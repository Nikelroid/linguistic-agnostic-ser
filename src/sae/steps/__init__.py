"""SAE study pipeline steps (Steps 0-8 of the SAE_CONTEXT brief).

Each module is runnable as ``python -m src.sae.steps.<module> [args]`` and is also
reachable through the unified dispatcher ``python -m src.sae.steps <command>``
(see ``__main__.py``). Steps consume the cached per-frame activations produced by
``src.sae.extraction`` and write committed artifacts under ``results/SAE/``.

Command -> module map:
    inventory          inventory                (EXP folder inventory)
    step1              step1_train_sae           (train a TopK SAE)
    step2              step2_emotion_features     (confound-controlled emotion features)
    step3              step3_acoustic_lexical     (acoustic-vs-lexical positive control)
    step4              step4_depth_sweep          (decode-vs-disentangle depth, Q i)
    step5              step5_cross_encoder         (cross-paradigm overlap, Q ii.1)
    step6              step6_cross_speaker         (cross-speaker LOSO, Q ii.3)
    step7              step7_ablation              (causal sufficiency, Q iii)
    step8              step8_emis_text_bias        (EMIS text-bias probe, Q iv)
    step8b             step8b_emis_layer_sweep     (EMIS text-bias across depth/TTS)
    summary            build_summary               (package results/SAE + summary)
"""

# command name -> module basename within this package (extract lives one level up)
STEP_MODULES = {
    "inventory": "src.sae.steps.inventory",
    "step1": "src.sae.steps.step1_train_sae",
    "step2": "src.sae.steps.step2_emotion_features",
    "step3": "src.sae.steps.step3_acoustic_lexical",
    "step4": "src.sae.steps.step4_depth_sweep",
    "step5": "src.sae.steps.step5_cross_encoder",
    "step6": "src.sae.steps.step6_cross_speaker",
    "step7": "src.sae.steps.step7_ablation",
    "step8": "src.sae.steps.step8_emis_text_bias",
    "step8b": "src.sae.steps.step8b_emis_layer_sweep",
    "summary": "src.sae.steps.build_summary",
    "extract": "src.sae.extraction",
}
