from qsft.query import get_bch_decoder
from qsft.qsft import transform
from qsft.input_signal_subsampled import SubsampledSignal


def sampling_strategy(sampling_function, max_b, n, sample_save_dir, t=5):
    bs = list(range(3, max_b + 1))
    query_args = {
        "query_method": "complex",
        "num_subsample": 3,
        "delays_method_source": "joint-coded",
        "subsampling_method": "qsft",
        "delays_method_channel": "identity-siso",
        "num_repeat": 1,
        "b": max(bs),
        "all_bs": bs,
        "t": t
    }
    return SubsampledSignal(func=sampling_function, n=n, q=2, query_args=query_args, folder=sample_save_dir)


def support_recovery(type, signal, b, t=5):
    qsft_args = {
        "num_subsample": 3,
        "num_repeat": 1,
        "reconstruct_method_source": "coded",
        "reconstruct_method_channel": "identity-siso" if type != "hard" else "identity",
        "b": b,
        "source_decoder": get_bch_decoder(signal.n, t, dectype="ml-soft-t2" if type != "hard" else "hard",
                                          chase_depth=2 * t),
        "peeling_method": "multi-detect",
        "noise_sd": 0,
        "regress": "lasso",
        "res_energy_cutoff": 0.9,
        "trap_exit": False,
        "report": True
    }
    return transform(signal, **qsft_args)
