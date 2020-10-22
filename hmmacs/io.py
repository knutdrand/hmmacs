import pandas as pd
def write_narrowpeak(scored_regions, f):
    regions_dict, score_dict, peaks_dict = scored_regions
    for chrom, regions in regions_dict.items():
        df = pd.DataFrame({"chrom": chrom,
                           "start": regions.starts,
                           "end": regions.ends,
                           "name": [f"{chrom}_{i}" for i, _ in enumerate(regions.starts)],
                           "score": score_dict[chrom],
                           "strand": ".",
                           "signalValue": "-1",
                           "pValue": "-1",
                           "qValue": "-1",
                           "peaks": peaks_dict[chrom]})
        df.to_csv(f, sep="\t", header=False, index=False)
