import pandas as pd
def write_narrowpeak(scored_regions, f):
    regions_dict, score_dict = scored_regions
    for chrom, regions in regions_dict.items():
        df = pd.DataFrame({"chrom": chrom,
                           "start": regions.starts,
                           "end": regions.ends,
                           "name": [f"{chrom}_{i}" for i, _ in enumerate(regions.starts)],
                           "score": score_dict[chrom]})
        df.to_csv(f, sep="\t", header=False, index=False)
