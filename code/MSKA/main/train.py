import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import aux.utils as utils
from aux.utils import EarlyStopping
from Recognition.Tokenizer import GlossTokenizer_S2G
import os
import time
import argparse, json, datetime
import numpy as np
from collections import defaultdict
import yaml
import random
import wandb
from pathlib import Path
import math
import sys
from typing import Iterable
from loguru import logger

# Métricas de evaluación:
#   wer_list → Word Error Rate (para reconocimiento de glosas, cuanto menor mejor)
#   bleu     → BLEU-1/2/3/4 (para traducción de texto, cuanto mayor mejor)
#   rouge    → ROUGE (para traducción de texto, cuanto mayor mejor)
from aux.metrics import wer_list, bleu, rouge
from aux.optimizer import build_optimizer, build_scheduler
# Funciones de limpieza de texto específicas de cada variante del dataset Phoenix
# (normalizan mayúsculas, puntuación, etc.) antes de calcular métricas
from aux.phoenix_cleanup import clean_phoenix_2014_trans, clean_phoenix_2014

import warnings
warnings.filterwarnings("ignore")

import csv

#TODO ICL bien hecho.

# ---------------------------------------------------------------------------
# Argumentos de línea de comandos
# ---------------------------------------------------------------------------

def get_args_parser():
    parser = argparse.ArgumentParser('VLP V2 scripts', add_help=False)

    # Entrenamiento básico
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--config', type=str, default='')

    # Gestión de checkpoints:
    # --finetune: carga pesos preentrenados de forma parcial (strict=False),
    #             útil para transfer learning donde el modelo puede tener capas distintas
    # --resume:   reanuda un entrenamiento interrumpido cargando modelo + optimizador + scheduler (strict=True)
    parser.add_argument('--finetune', default='', help='cargar pesos preentrenados (carga parcial)')
    parser.add_argument('--resume', default='', help='reanudar desde checkpoint (carga total)')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')

    # Modo evaluación: salta el entrenamiento y solo evalúa el modelo cargado con --resume
    parser.add_argument('--eval', action='store_true', help='solo evaluar, no entrenar')

    # Pin memory: mantiene los batches en RAM fija para que la
    # transferencia CPU→GPU sea más rápida. Activo por defecto.
    parser.add_argument('--pin-mem', action='store_true')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Parámetros de Weights & Biases
    parser.add_argument("--entity", type=str, help="entidad de wandb")
    parser.add_argument("--project", type=str, default='VLP', help="proyecto de wandb")

    parser.add_argument('--slm', action='store_true', help='Activa el modo SLM')

    return parser


# ---------------------------------------------------------------------------
# Función principal: prepara datos, modelo y lanza el bucle de entrenamiento
# ---------------------------------------------------------------------------

def main(args, config):
    print(args)
    device = torch.device(args.device)

    # Fijar semillas para reproducibilidad
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # benchmark=False garantiza resultados deterministas a costa de algo de velocidad
    cudnn.benchmark = False

    early_stopping = EarlyStopping(
    patience=config['training'].get('early_stopping_patience', 15),
    mode='min' if config['task'] == 'S2G' else 'max',
    min_delta=config['training'].get('early_stopping_min_delta', 0.0),
    )

    if args.slm:
        from slm.S2T_Dataset import S2T_Dataset
        from slm.model_slm import SignLanguageModel

    else:
        from MBart.datasets import S2T_Dataset
        from MBart.model_MBart import SignLanguageModel


    # --- Datasets y DataLoaders ---
    print("Creating dataset:")
    tokenizer = GlossTokenizer_S2G(config['gloss'])

    train_data = S2T_Dataset(path=config['data']['train_label_path'], tokenizer=tokenizer,
                             config=config, args=args, phase='train', training_refurbish=True)
    print(train_data)
    train_dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=train_data.collate_fn,
        shuffle=True,           # mezclar muestras cada época
        pin_memory=args.pin_mem,
    )

    dev_data = S2T_Dataset(path=config['data']['dev_label_path'], tokenizer=tokenizer,
                           config=config, args=args, phase='val', training_refurbish=True)
    print(dev_data)
    dev_dataloader = DataLoader(
        dev_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dev_data.collate_fn,
        pin_memory=args.pin_mem,
        # sin shuffle: en evaluación queremos ver todas las muestras en orden
    )

    test_data = S2T_Dataset(path=config['data']['test_label_path'], tokenizer=tokenizer,
                            config=config, args=args, phase='test', training_refurbish=True)
    print(test_data)
    test_dataloader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=test_data.collate_fn,
        pin_memory=args.pin_mem,
    )

    # --- Modelo ---
    print("Creating model:")
    model = SignLanguageModel(cfg=config, args=args)
    model.to(device)
    print(model)

    # Carga parcial de pesos preentrenados para fine-tuning.
    # strict=False permite que el checkpoint tenga capas que el modelo actual no tiene
    # (o viceversa), ignorando las incompatibles en lugar de lanzar un error.
    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        ret = model.load_state_dict(checkpoint['model'], strict=False)
        print('Missing keys: \n', '\n'.join(ret.missing_keys))
        print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))

    n_parameters = utils.count_parameters_in_MB(model)
    print(f'number of params: {n_parameters}M')

    optimizer = build_optimizer(config=config['training']['optimization'], model=model)
    scheduler, scheduler_type = build_scheduler(config=config['training']['optimization'], optimizer=optimizer)
    output_dir = Path(config['training']['model_dir'])

    # Reanudar entrenamiento interrumpido: restaura modelo, optimizador, scheduler y época
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
        # Solo restaurar optimizador/scheduler si estamos entrenando (no evaluando)
        if not args.eval and 'optimizer' in checkpoint and 'scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    # --- Modo evaluación (--eval): evaluar y salir sin entrenar ---
    if args.eval:
        if not args.resume:
            logger.warning('Please specify the trained model: --resume /path/to/best_checkpoint.pth')
        dev_stats = evaluate(args, config, dev_dataloader, model, tokenizer, epoch=0, beam_size=5,
                             generate_cfg=config['training']['validation']['translation'],
                             do_translation=config['do_translation'], do_recognition=config['do_recognition'])
        print(f"Dev loss of the network on the {len(dev_dataloader)} test videos: {dev_stats['loss']:.3f}")
        test_stats = evaluate(args, config, test_dataloader, model, tokenizer, epoch=0, beam_size=5,
                              generate_cfg=config['testing']['translation'],
                              do_translation=config['do_translation'], do_recognition=config['do_recognition'])
        print(f"Test loss of the network on the {len(test_dataloader)} test videos: {test_stats['loss']:.3f}")
        return

    # --- Bucle de entrenamiento ---
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    # Valores de referencia para decidir cuándo guardar el mejor checkpoint:
    # S2T usa BLEU-4 (maximizar), S2G usa WER (minimizar)
    min_wer = 200
    best_bleu4 = 0

    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()
        train_stats = train_one_epoch(args, model, tokenizer, train_dataloader, optimizer, device, epoch)

        # Guardar checkpoint de la época actual (se sobreescribe cada vez).
        # El optimizador está omitido intencionalmente para reducir el tamaño del fichero;
        # solo se guarda completo en best_checkpoint más abajo.
        torch.save({
            'model': model.state_dict(),
            'scheduler': scheduler.state_dict(),
            'optimizer': optimizer.state_dict(), 
            'epoch': epoch,
        }, output_dir / 'checkpoint.pth')

        # Evaluar en el conjunto de validación (dev) tras cada época
        test_stats = evaluate(args, config, dev_dataloader, model, tokenizer, epoch,
                              beam_size=config['training']['validation']['recognition']['beam_size'],
                              generate_cfg=config['training']['validation']['translation'],
                              do_translation=config['do_translation'], do_recognition=config['do_recognition'])

        # Guardar el mejor checkpoint según la métrica de la tarea
        best_checkpoint = {
            'model': model.state_dict(),
            #'optimizer': optimizer.state_dict(), # FIXME No guardo el optimizer pq andamos justo de espacio
            #'scheduler': scheduler.state_dict(), # FIXME No guardo el optimizer pq andamos justo de espacio
            'epoch': epoch,
        }
        if config['task'] == "S2T":
            # Traducción: guardar si mejora el BLEU-4
            if best_bleu4 < test_stats["bleu4"]:
                best_bleu4 = test_stats["bleu4"]
                torch.save(best_checkpoint, output_dir / 'best_checkpoint.pth')
            print(f"* DEV BLEU-4 {test_stats['bleu4']:.3f} Max DEV BLEU-4 {best_bleu4}")
        else:
            # Reconocimiento: guardar si baja el WER
            if min_wer > test_stats["wer"]:
                min_wer = test_stats["wer"]
                torch.save(best_checkpoint, output_dir / 'best_checkpoint.pth')
            print(f"* DEV wer {test_stats['wer']:.3f} Min DEV WER {min_wer}")

        # Loggear métricas en wandb
        if args.run:
            args.run.log({
                'epoch': epoch + 1,
                'training/train_loss': train_stats['loss'],
                'dev/dev_loss': test_stats['loss'],
                'dev/min_loss': min_wer,
            })

        # Guardar estadísticas de la época en log.txt
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters,
        }
        with (output_dir / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")

        metric = test_stats['wer'] if config['task'] == 'S2G' else test_stats['bleu4']
        if early_stopping(metric):
            print(f"Early stopping en época {epoch}. Mejor: {early_stopping.best:.4f}")
            break

    # --- Evaluación final con el mejor checkpoint ---
    # Al terminar el bucle, recargar el mejor modelo y evaluar en dev y test
    checkpoint = torch.load(str(output_dir) + '/best_checkpoint.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=True)

    dev_stats = evaluate(args, config, dev_dataloader, model, tokenizer, epoch=0,
                         beam_size=config['testing']['recognition']['beam_size'],
                         generate_cfg=config['training']['validation']['translation'],
                         do_translation=config['do_translation'], do_recognition=config['do_recognition'])
    print(f"Dev loss de la red en los {len(dev_dataloader)} test videos: {dev_stats['loss']:.3f}")

    test_stats = evaluate(args, config, test_dataloader, model, tokenizer, epoch=0,
                          beam_size=config['testing']['recognition']['beam_size'],
                          generate_cfg=config['testing']['translation'],
                          do_translation=config['do_translation'], do_recognition=config['do_recognition'])
    print(f"Test loss de la red en los {len(test_dataloader)} test videos: {test_stats['loss']:.3f}")

    with (output_dir / "log.txt").open("a") as f:
        if config['do_recognition']:
            f.write(json.dumps({'Dev WER:': dev_stats['wer'], 'Test WER:': test_stats['wer']}) + "\n")
        if config['do_translation']:
            f.write(json.dumps({'Dev Bleu-4:': dev_stats['bleu4'], 'Test Bleu-4:': test_stats['bleu4']}) + "\n")

    total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print('Training time {}'.format(total_time_str))


# ---------------------------------------------------------------------------
# Un paso completo de entrenamiento sobre todos los batches de una época
# ---------------------------------------------------------------------------

def train_one_epoch(args, model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int):
    model.train()  # activa dropout, batch norm en modo entrenamiento, etc.
    metric_logger = utils.MetricLogger(delimiter="  ")
    
    for group in optimizer.param_groups:
        group_name = group.get('name', 'default')
        metric_logger.add_meter(f'lr_{group_name}', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        
    header = f'Epoca: [{epoch}/{args.epochs}]'

    for step, (batch) in enumerate(metric_logger.log_every(data_loader, print_freq=10, header=header)):
        optimizer.zero_grad()        # limpiar gradientes del paso anterior

        output = model(batch)    # forward pass: calcula predicciones y loss
        

        # set_detect_anomaly detecta NaNs/Infs durante el backward e indica exactamente
        # dónde ocurrieron. Útil para depurar pero tiene coste de rendimiento;
        # se puede desactivar una vez el modelo sea estable.
        with torch.autograd.set_detect_anomaly(True):
            output['total_loss'].backward()  # backward pass: calcula gradientes

        optimizer.step()             # actualizar pesos con los gradientes calculados
        model.zero_grad()

        loss_value = output['total_loss'].item()

        # Si la loss es NaN o Inf el modelo ha divergido; no tiene sentido continuar
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        metric_logger.update(loss=loss_value)
        #metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        # Extraemos el lr actual de cada grupo del optimizador
        current_lrs = {}
        for group in optimizer.param_groups:
            # Usamos el nombre que le dimos en build_optimizer (si existe)
            group_name = group.get('name', 'default') 
            lr_val = group["lr"]
            current_lrs[f"lr_{group_name}"] = lr_val
            # Actualizamos el logger de texto en consola
            metric_logger.update(**{f"lr_{group_name}": lr_val})

        # --- Registrar los pesos en el metric_logger ---
        if 'stream_weights' in output:
            w = output['stream_weights']
            metric_logger.update(
                weight_left=w[0].item(),
                weight_right=w[1].item(),
                weight_body=w[2].item(),
                weight_fuse=w[3].item()
            )
        # ------------------------------------------------------
        if step % 50 == 0 and 'stream_weights' in output:
            w = output['stream_weights']
            print(f"Stream weights — left:{w[0]:.3f} right:{w[1]:.3f} body:{w[2]:.3f} fuse:{w[3]:.3f}")
            
        # if step == 10:
        #     print("\n"*2)
        #     for name, param in model.named_parameters():
        #         if param.requires_grad and param.grad is not None:
        #             print(f"{name}: grad_norm={param.grad.norm():.4f}")
        #     print("\n"*2)

    if args.run and 'stream_weights' in output:
        w = output['stream_weights']
        args.run.log({
            'epoch': epoch + 1,
            'stream_weights/left':  w[0].item(),
            'stream_weights/right': w[1].item(),
            'stream_weights/body':  w[2].item(),
            'stream_weights/fuse':  w[3].item(),
            'epoch/train_loss':     loss_value,
        })

    elif args.run:
        wandb_logs = {
            'epoch': epoch + 1,
            'epoch/train_loss': loss_value,}
        
        # Añadimos los LRs al diccionario de wandb
        for group_name, lr_val in current_lrs.items():
            wandb_logs[f'epoch/{group_name}'] = lr_val

        args.run.log(wandb_logs)


    print("Metricas medias:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ---------------------------------------------------------------------------
# Evaluación: inferencia sin gradientes + cálculo de métricas
# ---------------------------------------------------------------------------

def evaluate(args, config, dev_dataloader, model, tokenizer, epoch, beam_size=1,
             generate_cfg={}, do_translation=True, do_recognition=True):
    model.eval()  # desactiva dropout, batch norm en modo inferencia, etc.
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # Acumulamos predicciones y referencias por nombre de muestra para
    # calcular las métricas globales al final (no batch a batch)
    results = defaultdict(dict)

    with torch.no_grad():  # desactiva el cálculo de gradientes para ahorrar memoria y tiempo
        for step, (src_input) in enumerate(metric_logger.log_every(dev_dataloader, print_freq=10, header=header)):
            output = model(src_input)

            # --- Reconocimiento de glosas (S2G) ---
            # El modelo puede tener varias cabezas de clasificación (distintos 'gloss_logits_*').
            # Se evalúan todas y se conserva la que tenga menor WER.
            if do_recognition:
                for k, gls_logits in output.items():
                    if 'gloss_logits' not in k:
                        continue
                    logits_name = k.replace('gloss_logits', '')

                    # CTC decode: convierte logits en secuencias de glosas usando beam search
                    ctc_decode_output = model.recognition_network.decode(
                        gloss_logits=gls_logits,
                        beam_size=beam_size,
                        input_lengths=output['input_lengths'],
                    )
                    # Convertir IDs numéricos a tokens de texto (glosas)
                    batch_pred_gls = tokenizer.convert_ids_to_tokens(ctc_decode_output)

                    for name, gls_hyp, gls_ref in zip(src_input['name'], batch_pred_gls, src_input['gloss']):
                        # Normalizar a mayúsculas si el tokenizador usa lower_case
                        results[name][f'{logits_name}gls_hyp'] = (
                            ' '.join(gls_hyp).upper() if tokenizer.lower_case else ' '.join(gls_hyp)
                        )
                        results[name]['gls_ref'] = (
                            gls_ref.upper() if tokenizer.lower_case else gls_ref
                        )

            # --- Traducción de texto (S2T) ---
            # El transformer genera texto con beam search a partir de las features del encoder
            if do_translation:
                generate_output = model.generate_txt(
                    transformer_inputs=output['transformer_inputs'],
                    generate_cfg=generate_cfg,
                )
                #print(generate_output)
                # --- INICIO DEL REGISTRO EN CSV ---
#                 # Definir la ruta del archivo (se guardará en la carpeta actual o puedes especificar una ruta)
#                 csv_file = 'predictions_sample_20.csv'
                
#                 # Si el archivo no existe, creamos los encabezados
#                 file_exists = os.path.isfile(csv_file)
#                 with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
#                     writer = csv.writer(f)
#                     if not file_exists:
#                         writer.writerow(['Video_Name', 'Ground_Truth', 'Prediction'])
                    
#                     # Iterar sobre el batch y guardar en el diccionario y en el CSV
#                     for name, txt_hyp, txt_ref in zip(src_input['name'],
#                                                       generate_output['decoded_sequences'],
#                                                       src_input['text']):
#                         results[name]['txt_hyp'] = txt_hyp
#                         results[name]['txt_ref'] = txt_ref
                        
#                         # Escribir la fila en el CSV
#                         writer.writerow([name, txt_ref, txt_hyp])
# #                 # --- FIN DEL REGISTRO EN CSV ---
                
                for name, txt_hyp, txt_ref in zip(src_input['name'],
                                                   generate_output['decoded_sequences'],
                                                   src_input['text']):
                    results[name]['txt_hyp'] = txt_hyp
                    results[name]['txt_ref'] = txt_ref

            metric_logger.update(loss=output['total_loss'].item())

        # --- Calcular WER global sobre todas las muestras ---
        if do_recognition:
            evaluation_results = {'wer': 200}  # 200 como valor imposible (WER máximo real es 100%)

            for hyp_name in results[name].keys():
                if 'gls_hyp' not in hyp_name:
                    continue
                k = hyp_name.replace('gls_hyp', '')

                # Limpiar texto según el formato específico de cada dataset antes de comparar
                dataset = config['data']['dataset_name'].lower()
                if dataset == 'phoenix-2014t':
                    gls_ref = [clean_phoenix_2014_trans(results[n]['gls_ref']) for n in results]
                    gls_hyp = [clean_phoenix_2014_trans(results[n][hyp_name]) for n in results]
                else:
                    raise NotImplementedError()

                wer_results = wer_list(hypotheses=gls_hyp, references=gls_ref)
                evaluation_results[k + 'wer_list'] = wer_results
                # Conservar el WER de la mejor cabeza de clasificación
                evaluation_results['wer'] = min(wer_results['wer'], evaluation_results['wer'])

            metric_logger.update(wer=evaluation_results['wer'])

        # --- Calcular BLEU y ROUGE global sobre todas las muestras ---
        if do_translation:
            txt_ref = [results[n]['txt_ref'] for n in results]
            txt_hyp = [results[n]['txt_hyp'] for n in results]

            bleu_dict = bleu(references=txt_ref, hypotheses=txt_hyp, level=config['data']['level'])
            rouge_score = rouge(references=txt_ref, hypotheses=txt_hyp, level=config['data']['level'])

            for k, v in bleu_dict.items():
                print(f'{k} {v:.2f}')
            print(f'ROUGE: {rouge_score:.2f}')

            evaluation_results['rouge'] = rouge_score
            evaluation_results['bleu'] = bleu_dict

            wandb.log({'eval/BLEU4': bleu_dict['bleu4'], 'eval/ROUGE': rouge_score})
            metric_logger.update(bleu1=bleu_dict['bleu1'], bleu2=bleu_dict['bleu2'],
                                  bleu3=bleu_dict['bleu3'], bleu4=bleu_dict['bleu4'],
                                  rouge=rouge_score)

    if args.run:
        args.run.log({
            'epoch': epoch + 1,
            'epoch/dev_loss': output['recognition_loss'].item(),
            'wer': evaluation_results['wer'],
        })



    print("* Averaged stats:", metric_logger)
    #print('* DEV loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))
    print('* DEV loss {losses.global_avg:.3f}'.format(losses=metric_logger.meters['loss']))
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ---------------------------------------------------------------------------
# Configuración de Weights & Biases
# ---------------------------------------------------------------------------

def setup_run(args, config):
    """
    Inicializa un run de wandb para registrar métricas del experimento.
    Si args.eval está activo, wandb se deshabilita (no tiene sentido loggear evaluaciones).
    """
    os.environ["WANDB_MODE"] = config['training']['wandb'] if not args.eval else 'disabled'

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        config=config,
    )
    run.define_metric("epoch")
    run.define_metric("training/*", step_metric="epoch")
    run.define_metric("dev/*", step_metric="epoch")
    return run


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Evitar deadlocks en tokenizadores HuggingFace cuando se usan múltiples workers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('VLP V2 scripts', parents=[get_args_parser()])
    args = parser.parse_args()

    with open(args.config, 'r+', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    args.run = setup_run(args, config)

    # Crear directorio de salida si no existe
    Path(config['training']['model_dir']).mkdir(parents=True, exist_ok=True)

    main(args, config)
