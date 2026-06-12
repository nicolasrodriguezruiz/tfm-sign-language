# coding: utf-8
"""
Constructores de optimizador y scheduler de learning rate.
"""
from typing import Callable, Optional, Generator

import torch
from torch import nn
from torch.optim import lr_scheduler, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import warnings
import math


# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------

def build_gradient_clipper(config: dict) -> Optional[Callable]:
    """
    Construye una función de gradient clipping según el config.

    El gradient clipping evita el problema de "exploding gradients": cuando
    los gradientes crecen demasiado durante el backward, los pesos se actualizan
    con pasos enormes y el entrenamiento diverge. Hay dos variantes:

      - 'clip_grad_val': recorta cada gradiente individualmente si supera clip_value.
            Ejemplo: gradiente de 5.0 con clip_value=1.0 → se recorta a 1.0.

      - 'clip_grad_norm': recorta todos los gradientes proporcionalmente si su
            norma L2 global supera max_norm. Preserva la dirección del gradiente,
            solo reduce su magnitud.
            Ejemplo: norma=10, max_norm=1 → todos los gradientes se dividen por 10.

    Si no se especifica ninguno, devuelve None (sin clipping).
    No se pueden usar los dos a la vez.

    Returns:
        Función que recibe los parámetros del modelo y aplica el clipping in-place,
        o None si no se especifica clipping.
    """
    if "clip_grad_val" in config and "clip_grad_norm" in config:
        raise ValueError("Solo se puede especificar clip_grad_val o clip_grad_norm, no ambos.")

    if "clip_grad_val" in config:
        clip_value = config["clip_grad_val"]
        return lambda params: nn.utils.clip_grad_value_(
            parameters=params, clip_value=clip_value
        )
    elif "clip_grad_norm" in config:
        max_norm = config["clip_grad_norm"]
        return lambda params: nn.utils.clip_grad_norm_(
            parameters=params, max_norm=max_norm
        )
    return None


# ---------------------------------------------------------------------------
# Optimizador con learning rates por módulo
# ---------------------------------------------------------------------------


def build_optimizer(config: dict, model) -> Optimizer:
    optimizer_name = config.get("optimizer", "adam").lower()
    weight_decay   = config.get("weight_decay", 0.00)
    eps            = config.get("eps", 1.0e-8)
    betas          = config.get("betas", (0.9, 0.999))
    amsgrad        = config.get("amsgrad", False)

    base_lr = config['learning_rate'].pop('default')
    lr_map  = config['learning_rate']   # resto de claves después del pop

    # Palabras clave de parámetros que NO deben llevar weight decay
    # Atrapamos sesgos (bias) y capas de normalización (LayerNorm, BatchNorm)
    no_decay_keywords = ['bias', 'LayerNorm', 'BatchNorm', 'GroupNorm']

    # Crear listas separadas: Módulo (LR) + Subgrupo (Decay / No Decay)
    groups = {
        'lora':           {'params': [], 'lr': lr_map.get('lora', base_lr),        'weight_decay': weight_decay},
        'lora_nd':        {'params': [], 'lr': lr_map.get('lora', base_lr),        'weight_decay': 0.0},
        'mapper':         {'params': [], 'lr': lr_map.get('mapper', base_lr),      'weight_decay': weight_decay},
        'mapper_nd':      {'params': [], 'lr': lr_map.get('mapper', base_lr),      'weight_decay': 0.0},
        'recognition':    {'params': [], 'lr': lr_map.get('recognition', base_lr), 'weight_decay': weight_decay},
        'recognition_nd': {'params': [], 'lr': lr_map.get('recognition', base_lr), 'weight_decay': 0.0},
        'default':        {'params': [], 'lr': base_lr,                            'weight_decay': weight_decay},
        'default_nd':     {'params': [], 'lr': base_lr,                            'weight_decay': 0.0}
    }

    # Agrupar parámetros por nombre para asignar lr y decay individualmente.
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Comprobar si el parámetro debe ignorar el weight_decay
        has_no_decay = any(nd in name for nd in no_decay_keywords)
        suffix = '_nd' if has_no_decay else ''

        if 'lora_' in name or 'TranslationNetwork' in name:
            groups['lora' + suffix]['params'].append(param)
        elif 'vl_mapper' in name or 'VLMapper' in name:
            groups['mapper' + suffix]['params'].append(param)
        elif 'recognition_network' in name or 'RecognitionNetwork' in name:
            groups['recognition' + suffix]['params'].append(param)
        else:
            groups['default' + suffix]['params'].append(param)

    # Filtrar grupos vacíos (por si acaso) y preparar la lista final para PyTorch
    parameters = []
    for group_name, group_data in groups.items():
        if len(group_data['params']) > 0:
            parameters.append({
                'params': group_data['params'],
                'lr': group_data['lr'],
                'weight_decay': group_data['weight_decay'],
                # Quitamos el sufijo '_nd' del nombre al guardarlo.
                # Así tu train.py seguirá logueando en consola y en WandB
                # un único y limpio 'lr_recognition' en lugar de duplicarlo.
                'name': group_name.replace('_nd', '')
            })

    if optimizer_name == "adam":
        return torch.optim.Adam(
            params=parameters, lr=base_lr, betas=betas,
            eps=eps, weight_decay=weight_decay, amsgrad=amsgrad,
        )
    elif optimizer_name == "adamw":
        # AdamW: igual que Adam pero con weight decay desacoplado.
        # En Adam estándar el weight decay interactúa con la adaptación del lr,
        # lo que reduce su efecto real. AdamW lo aplica directamente a los pesos.
        # PyTorch respetará el 'weight_decay' específico que hemos metido
        # dentro de cada diccionario en `parameters` (ignorando el global para los subgrupos _nd).
        return torch.optim.AdamW(
            params=parameters, lr=base_lr, betas=betas,
            eps=eps, weight_decay=weight_decay, amsgrad=amsgrad,
        )
    elif optimizer_name == "adagrad":
        # Adagrad: acumula el cuadrado de los gradientes históricos y divide por su raíz.
        # Los parámetros que reciben gradientes grandes se actualizan menos con el tiempo.
        # Útil para datos dispersos, pero el lr puede decrecer demasiado en entrenamientos largos.
        return torch.optim.Adagrad(
            params=parameters, lr=base_lr,
            lr_decay=config.get("lr_decay", 0),
            weight_decay=weight_decay, eps=eps,
        )
    elif optimizer_name == "adadelta":
        # Adadelta: mejora de Adagrad que usa una ventana deslizante en lugar de acumulación total.
        # rho: factor de decaimiento de la ventana (similar a momentum).
        return torch.optim.Adadelta(
            params=parameters, rho=config.get("rho", 0.9),
            eps=eps, lr=base_lr, weight_decay=weight_decay,
        )
    elif optimizer_name == "rmsprop":
        # RMSProp: divide el gradiente por la raíz de la media móvil de su cuadrado.
        # Estabiliza el entrenamiento en redes recurrentes.
        # alpha: factor de decaimiento de la media móvil.
        return torch.optim.RMSprop(
            params=parameters, lr=base_lr,
            momentum=config.get("momentum", 0),
            alpha=config.get("alpha", 0.99),
            eps=eps, weight_decay=weight_decay,
        )
    elif optimizer_name == "sgd":
        # SGD: el más simple. Actualiza pesos en la dirección del gradiente.
        # Con momentum acumula velocidad en direcciones consistentes,
        # reduciendo oscilaciones y acelerando la convergencia.
        return torch.optim.SGD(
            params=parameters, lr=base_lr,
            momentum=config.get("momentum", 0),
            weight_decay=weight_decay,
        )
    else:
        raise ValueError("Optimizador desconocido: {}.".format(optimizer_name))

# ---------------------------------------------------------------------------
# Schedulers de learning rate
# ---------------------------------------------------------------------------

def build_scheduler(config: dict, optimizer: Optimizer,
                    scheduler_mode: str = 'max', hidden_size: int = 0):
    """
    Construye el scheduler de learning rate según el config.

    Un scheduler modifica el lr durante el entrenamiento. Empezar con un lr
    alto permite explorar el espacio de parámetros; reducirlo después permite
    converger a un mínimo más preciso.

    Returns:
        (scheduler, scheduler_step_at): el scheduler y cuándo llamar a .step():
            - 'epoch':      al final de cada época
            - 'validation': después de cada evaluación (para plateau)
            - 'step':       después de cada batch
    """
    scheduler_name = config["scheduler"].lower()

    if scheduler_name == "plateau":
        # Reduce el lr cuando la métrica de validación deja de mejorar.
        # patience: número de evaluaciones sin mejora antes de reducir el lr.
        # factor: multiplicador del lr cuando se activa (nuevo_lr = lr * factor).
        # Útil cuando no se sabe cuándo el modelo dejará de mejorar.
        return (
            lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode=scheduler_mode,        # 'max' para BLEU, 'min' para WER/loss
                verbose=False,
                threshold_mode="abs",
                factor=config.get("decrease_factor", 0.1),
                patience=config.get("patience", 10),
            ),
            "validation",
        )

    elif scheduler_name == "cosineannealing":
        # El lr sigue una curva coseno: decrece suavemente desde lr_max hasta eta_min.
        # T_max: número de épocas para completar medio ciclo coseno.
        # Más suave que StepLR (sin caídas abruptas).
        return (
            lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                eta_min=config.get("eta_min", 0),
                T_max=config.get("t_max", 20),
            ),
            "epoch",
        )

    elif scheduler_name == 'warmup_cosineannealing':
        # Cosine annealing con warmup. Pendiente de implementar (devuelve None).
        # Ver la clase WarmupCosineannealing comentada al final del fichero.
        return None

    elif scheduler_name == "cosineannealingwarmrestarts":
        # Como CosineAnnealing pero reinicia el ciclo periódicamente.
        # Permite al modelo escapar de mínimos locales cuando el lr "sube" de nuevo.
        # T_0: duración del primer ciclo en épocas.
        # T_mult: factor por el que se alarga cada ciclo siguiente (T_mult=2 → 10, 20, 40...).
        return (
            lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=config.get("t_init", 10),
                T_mult=config.get("t_mult", 2),
            ),
            "step",
        )

    elif scheduler_name == "decaying":
        # StepLR: multiplica el lr por gamma cada decaying_step_size épocas.
        # Simple y predecible, pero la caída es abrupta.
        return (
            lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=config.get("decaying_step_size", 1),
            ),
            "epoch",
        )

    elif scheduler_name == "exponential":
        # Multiplica el lr por gamma en cada época: lr = lr * gamma^época.
        # Decaimiento suave y continuo.
        return (
            lr_scheduler.ExponentialLR(
                optimizer=optimizer,
                gamma=config.get("decrease_factor", 0.99),
            ),
            "epoch",
        )

    elif scheduler_name == "noam":
        # Scheduler del paper "Attention is All You Need".
        # Sube linealmente durante warmup steps y luego decae como 1/sqrt(step).
        # Ver NoamScheduler para más detalle.
        return (
            NoamScheduler(
                hidden_size=hidden_size,
                factor=config.get("learning_rate_factor", 1),
                warmup=config.get("learning_rate_warmup", 4000),
                optimizer=optimizer,
            ),
            "step",
        )

    elif scheduler_name == "warmupexponentialdecay":
        # Warmup lineal seguido de decaimiento exponencial con lr mínimo garantizado.
        # Ver WarmupExponentialDecayScheduler para más detalle.
        return (
            WarmupExponentialDecayScheduler(
                min_rate=config.get("learning_rate_min", 1.0e-5),
                decay_rate=config.get("learning_rate_decay", 0.1),
                warmup=config.get("learning_rate_warmup", 4000),
                optimizer=optimizer,
                peak_rate=config.get("learning_rate_peak", 1.0e-3),
                decay_length=config.get("learning_rate_decay_length", 10000),
            ),
            "step",
        )

    else:
        raise ValueError("Scheduler desconocido: {}.".format(scheduler_name))


# ---------------------------------------------------------------------------
# Scheduler Noam (del paper "Attention is All You Need")
# ---------------------------------------------------------------------------

class NoamScheduler:
    """
    Scheduler del paper original del Transformer (Vaswani et al., 2017).
    https://arxiv.org/pdf/1706.03762.pdf  Eq. 3

    Fórmula:
        lr = factor * hidden_size^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))

    Comportamiento:
        - Durante warmup: lr crece linealmente (step * warmup^(-1.5) domina).
        - Después de warmup: lr decae como 1/sqrt(step).

    La escala hidden_size^(-0.5) ajusta el lr al tamaño del modelo:
    modelos más grandes usan un lr más pequeño de partida.

    Visualización:
        lr
        ▲
        │     /\\
        │    /  \\
        │   /    \\____
        │  /          \\___
        └─────────────────► steps
              ↑
           warmup
    """

    def __init__(self, hidden_size: int, optimizer: torch.optim.Optimizer,
                 factor: float = 1, warmup: int = 4000):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.hidden_size = hidden_size
        self._rate = 0

    def step(self):
        """Avanza un paso y actualiza el lr en todos los grupos del optimizador."""
        self._step += 1
        rate = self._compute_rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate

    def _compute_rate(self):
        step = self._step
        return self.factor * (
            self.hidden_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5))
        )

    def state_dict(self):
        # No guarda estado (el scheduler se puede reconstruir desde el config)
        return None


# ---------------------------------------------------------------------------
# Scheduler con warmup + decaimiento exponencial
# ---------------------------------------------------------------------------

class WarmupExponentialDecayScheduler:
    """
    Variante del scheduler Noam con decaimiento exponencial en lugar de 1/sqrt.

    Permite controlar la velocidad de decaimiento con decay_rate y garantiza
    un lr mínimo (min_rate) para que el entrenamiento nunca se detenga del todo.

    Fórmula:
        - Durante warmup:    lr = step * peak_rate / warmup
        - Después de warmup: lr = max(peak_rate * decay_rate^((step-warmup)/decay_length), min_rate)

    Visualización:
        lr
        ▲
        │     /\\
        │    /  \\
        │   /    \\
        │  /      \\___________  ← min_rate
        └─────────────────────► steps
              ↑
           warmup
    """

    def __init__(self, optimizer: torch.optim.Optimizer, peak_rate: float = 1.0e-3,
                 decay_length: int = 10000, warmup: int = 4000,
                 decay_rate: float = 0.5, min_rate: float = 1.0e-5):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.decay_length = decay_length
        self.peak_rate = peak_rate
        self._rate = 0
        self.decay_rate = decay_rate
        self.min_rate = min_rate

    def step(self):
        """Avanza un paso y actualiza el lr en todos los grupos del optimizador."""
        self._step += 1
        rate = self._compute_rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate

    def _compute_rate(self):
        step = self._step
        if step < self.warmup:
            # Fase de warmup: subida lineal desde 0 hasta peak_rate
            rate = step * self.peak_rate / self.warmup
        else:
            # Fase de decaimiento exponencial
            exponent = (step - self.warmup) / self.decay_length
            rate = self.peak_rate * (self.decay_rate ** exponent)
        # Garantizar que el lr nunca baje de min_rate
        return max(rate, self.min_rate)

    def state_dict(self):
        return None


# ---------------------------------------------------------------------------
# Scheduler con warmup lineal simple
# ---------------------------------------------------------------------------

class WarmupScheduler(_LRScheduler):
    """
    Scheduler simple que sube el lr linealmente durante total_epochs épocas.

    Útil como fase inicial antes de pasar a otro scheduler.
    Empieza en lr/total_epochs (no en 0) para evitar pasos demasiado pequeños
    en la primera época.

    Visualización:
        lr
        ▲         /
        │        /
        │       /
        │      /
        │     /
        │    /
        └───────────► epochs
            total_epochs
    """

    def __init__(self, optimizer, total_epochs, last_epoch=-1):
        self.total_epochs = total_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= 0:
            # Primera época: lr mínimo (1/total_epochs del base_lr)
            return [base_lr / (self.total_epochs + 1e-8) for base_lr in self.base_lrs]
        else:
            # Subida lineal: lr = base_lr * epoch / total_epochs
            return [base_lr * self.last_epoch / (self.total_epochs + 1e-8)
                    for base_lr in self.base_lrs]

    def finish(self):
        """Devuelve True cuando el warmup ha terminado."""
        return self.last_epoch >= self.total_epochs


# ---------------------------------------------------------------------------
# WarmupCosineannealing (pendiente de implementar)
# ---------------------------------------------------------------------------
# Combinaría la subida lineal de WarmupScheduler con el decaimiento coseno
# de CosineAnnealingLR. Está comentada porque la implementación del bloque
# de warmup (elif self.last_epoch < self.warmup) está incompleta.

# class WarmupCosineannealing(LRScheduler):
#     #based on https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#CosineAnnealingLR
#     def __init__(self, optimizer, T_max, warmup=0, eta_min=0, last_epoch=-1, verbose=False):
#         self.T_max = T_max
#         self.warmup = warmup #epoch
#         self.eta_min = eta_min
#         super(WarmupCosineannealing, self).__init__(optimizer, last_epoch, verbose)

#     def get_lr(self):
#         # add warmup (to-do)
#         if not self._get_lr_called_within_step:
#             warnings.warn("To get the last learning rate computed by the scheduler, "
#                           "please use `get_last_lr()`.", UserWarning)

#         if self.last_epoch == 0:
#             if self.warmup==0: #no warmup
#                 return [group['lr'] for group in self.optimizer.param_groups]
#             else:
#                 return [0 for group in self.optimizer.param_groups]
#         elif self.last_epoch < self.warmup:
#             #warmup

#         elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
#             return [group['lr'] + (base_lr - self.eta_min) *
#                     (1 - math.cos(math.pi / self.T_max)) / 2
#                     for base_lr, group in
#                     zip(self.base_lrs, self.optimizer.param_groups)]
#         return [(1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
#                 (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
#                 (group['lr'] - self.eta_min) + self.eta_min
#                 for group in self.optimizer.param_groups]

#     def _get_closed_form_lr(self):
#         #add warmup (to-do)
#         return [self.eta_min + (base_lr - self.eta_min) *
#                 (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
#                 for base_lr in self.base_lrs]

