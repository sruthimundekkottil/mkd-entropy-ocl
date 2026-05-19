from src.learners.baselines.agem import AGEMLearner
from src.learners.baselines.er import ERLearner
from src.learners.baselines.ocm import OCMLearner
from src.learners.ema.ocm_ema import OCMEMALearner
from src.learners.ce import CELearner
from src.learners.baselines.derpp import DERppLearner
from src.learners.ema.derpp_ema import DERppEMALearner
from src.learners.baselines.er_ace import ER_ACELearner
from src.learners.baselines.dvc import DVCLearner
from src.learners.ema.dvc_ema import DVCEMALearner
from src.learners.ema.er_ema import ER_EMALearner
from src.learners.ema.er_ace_ema import ER_ACE_EMALearner
from src.learners.baselines.gsa import GSALearner
from src.learners.ema.gsa_ema import GSAEMALearner
from src.learners.sdp.er_sdp import ER_SDPLearner
from src.learners.sdp.gsa_sdp import GSA_SDPLearner
from src.learners.sdp.er_ace_sdp import ER_ACE_SDPLearner
from src.learners.sdp.dvc_sdp import DVC_SDPLearner
from src.learners.sdp.derpp_sdp import DERpp_SDPLearner
from src.learners.baselines.pcr import PCRLearner
from src.learners.ema.pcr_ema import PCR_EMALearner
from src.learners.er_kdu import ER_KDULearner
from src.learners.ema.tens import TEnsLearner

# ── New learners added for this project ───────────────────────────────
from src.learners.baselines.er_entropy import ER_EntropyLearner
from src.learners.ema.er_ema_entropy import ER_EMA_EntropyLearner
from src.learners.baselines.vr_ocl import (
    VROCLLearner,
    VROCLDecayLearner,
)
from src.learners.baselines.vr_ocl_adaptive import VROCLAdaptiveLearner
from src.learners.baselines.ewc import EWCLearner
# ────────────────────────────────────────────────────────────────────

from src.buffers.reservoir import Reservoir
from src.buffers.protobuf import ProtoBuf
from src.buffers.SVDbuf import SVDbuf
from src.buffers.greedy import GreedySampler
from src.buffers.fifo import QueueMemory
from src.buffers.boostedbuf import BoostedBuffer
from src.buffers.mlbuf import MLBuf
from src.buffers.indexed_reservoir import IndexedReservoir
from src.buffers.logits_res import LogitsRes
from src.buffers.mgi_reservoir import MGIReservoir
from src.buffers.entropy_reservoir import EntropyReservoir


learners = {
    'ER':           ERLearner,
    'CE':           CELearner,
    'AGEM':         AGEMLearner,
    'OCM':          OCMLearner,
    'OCMEMA':       OCMEMALearner,
    'DERpp':        DERppLearner,
    'DERppEMA':     DERppEMALearner,
    'ERACE':        ER_ACELearner,
    'ERACE_EMA':    ER_ACE_EMALearner,
    'DVC':          DVCLearner,
    'DVC_EMA':      DVCEMALearner,
    'ER_EMA':       ER_EMALearner,
    'GSA':          GSALearner,
    'GSA_EMA':      GSAEMALearner,
    'ER_SDP':       ER_SDPLearner,
    'GSA_SDP':      GSA_SDPLearner,
    'ER_ACE_SDP':   ER_ACE_SDPLearner,
    'DVC_SDP':      DVC_SDPLearner,
    'DERpp_SDP':    DERpp_SDPLearner,
    'PCR':          PCRLearner,
    'PCR_EMA':      PCR_EMALearner,
    'ER_KDU':       ER_KDULearner,
    'TEns':         TEnsLearner,
    # ── New ──────────────────────────────────────────────────────────
    'ER_Entropy':       ER_EntropyLearner,       # ER + entropy replay
    'ER_EMA_Entropy':   ER_EMA_EntropyLearner,   # ER + MKD + entropy replay
    'VR_OCL':           VROCLLearner,            # variance regularizer fixed mu
    'VR_OCL_Decay':     VROCLDecayLearner,       # variance regularizer decaying mu
    'VR_OCL_Adaptive':  VROCLAdaptiveLearner,    # variance regularizer adaptive mu
    'EWC':              EWCLearner,              # EWC baseline
}

buffers = {
    'reservoir':        Reservoir,
    'protobuf':         ProtoBuf,
    'svd':              SVDbuf,
    'greedy':           GreedySampler,
    'logits_res':       LogitsRes,
    'fifo':             QueueMemory,
    'boost':            BoostedBuffer,
    'mlbuf':            MLBuf,
    'idx_reservoir':    IndexedReservoir,
    'mgi_reservoir':    MGIReservoir,
    'entropy_reservoir': EntropyReservoir,       # new
}
