import torch
import yaml
import gc
import os
import pandas as pd
from tqdm import tqdm
import evaluate
from torch.utils.data import DataLoader

from slm.S2T_Dataset import S2T_Dataset
from slm.model_slm import SignLanguageModel
from Recognition.Tokenizer import GlossTokenizer_S2G

import warnings
warnings.filterwarnings("ignore")

class DummyArgs:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seed = 0
        self.num_workers = 4
        self.pin_mem = True
        self.batch_size = 16
        self.slm = False

def main():
    config_path = 'configs/abalation/SLT_SLM_LoRA_Att_Qwen7_Gloss.yaml'
    checkpoint_path = '/home/user/work/data/outputs_final/SLT_SLM_Att_7B/best_checkpoint.pth'
    csv_output_path = 'resultados_test_s2t.csv'

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    args = DummyArgs()
    device = torch.device(args.device)

    # =====================================================================
    # INFERENCIA (QWEN) -> Extraer Traducciones
    # =====================================================================
    print("\n" + "="*50)
    print("INFERENCIA S2T CON QWEN")
    print("="*50)

    # 1. Cargar Datos
    print("[1/3] Cargando Tokenizador y DataLoader (Test Set)")
    tokenizer = GlossTokenizer_S2G(config['gloss'])

    test_data = S2T_Dataset(
        path=config['data']['test_label_path'],
        tokenizer=tokenizer,
        config=config, args=args, phase='test', training_refurbish=False
    )

    test_dataloader = DataLoader(
        test_data, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=test_data.collate_fn, shuffle=False, pin_memory=args.pin_mem
    )

    # 2. Cargar Modelo
    print("[2/3] Cargando Modelo de Traducción")
    model = SignLanguageModel(cfg=config, args=args)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['model'], strict=False)
    model.to(device)
    model.eval()

    # 3. Bucle de Inferencia
    print(f"[3/3] Generando traducciones para {len(test_data)} vídeos")
    resultados = []

    # Extraemos la config de generación (beam_size, max_new_tokens, etc.)
    gen_cfg = config['testing'].get('translation', {})

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Traduciendo"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward pass para obtener features del encoder
            output = model(batch)

            # Generar texto
            generate_output = model.generate_txt(
                transformer_inputs=output['transformer_inputs'],
                generate_cfg=gen_cfg,
            )

            # Guardar predicciones vs Ground Truth
            for name, txt_hyp, txt_ref in zip(batch['name'], generate_output['decoded_sequences'], batch['text']):
                resultados.append({
                    "Video_ID": name,
                    "Referencia": txt_ref,
                    "Prediccion": txt_hyp
                })

    # 4. Guardar a CSV
    df = pd.DataFrame(resultados)
    df.to_csv(csv_output_path, index=False, encoding='utf-8')
    print(f"\n Predicciones guardadas exitosamente en: {csv_output_path}")


if __name__ == "__main__":
    main()
