from sparse_transform.qsft.qsft import transform
from sparse_transform.qsft.codes.BCH import BCH
from sparse_transform.qsft.signals.input_signal_subsampled import SubsampledSignal
from functools import partial

def get_num_samples(signal, b):
    # Calculates the number of samples taken for the given sparsity parameter b
    return len(signal.all_samples) * len(signal.all_samples[0]) * (2 ** b)

def sampling_strategy(sampling_function, max_b, n, sample_save_dir, t=5):
    # takes samples for SPEX using BCH code defined in sparse_transform.qsft
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
    signal = SubsampledSignal(func=sampling_function, n=n, q=2, query_args=query_args, folder=sample_save_dir)
    num_samples = [get_num_samples(signal, b) for b in bs]
    return signal, num_samples


def support_recovery(type, signal, b, t=5):
    # Performs support recovery using decoding methods in sparse_transform.qsft
    # hard decoding is quicker, but soft decoding gives better recovery
    if type == "hard":
        source_decoder = BCH(signal.n, t).parity_decode
    else:
        source_decoder = partial(BCH(signal.n, t).parity_decode_2chase_t2_max_likelihood,
                                chase_depth=2*t)
    qsft_args = {
        "num_subsample": 3,
        "num_repeat": 1,
        "reconstruct_method_source": "coded",
        "reconstruct_method_channel": "identity-siso" if type != "hard" else "identity",
        "b": b,
        "source_decoder": source_decoder,
        "peeling_method": "multi-detect",
        "noise_sd": 0,
        "regress": 'lasso',
        "res_energy_cutoff": 0.9,
        "trap_exit": True,
        "verbosity": 0,
        "report": True,
        "peel_average": True,
    }
    return transform(signal, **qsft_args)
