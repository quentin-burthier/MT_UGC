"""Script to filter sentences in the wrong languages."""

def main(src_path: str, tgt_path: str,
         src_lid_path: str, tgt_lid_path: str,
         src_filtered: str, tgt_filtered: str,
         src_lang: str, tgt_lang: str):
    with open(src_path, "r") as src_raw, open(src_lid_path) as src_lid, open(tgt_path, "r") as tgt_raw, open(tgt_lid_path) as tgt_lid, open(src_filtered, "w") as src_f, open(tgt_filtered, "w") as tgt_f:
        for src_line, src_line_lang, tgt_line, tgt_line_lang in zip(src_raw, src_lid, tgt_raw, tgt_lid):
            if src_line_lang[-3:-1] == src_lang and tgt_line_lang[-3:-1] == tgt_lang:
                src_f.write(src_line)
                tgt_f.write(tgt_line)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 9:
        print("python $lid_filter.py {{$input_dir/,$filtered_dir/{labels.,}}train.,}{$src,$tgt}")
    else:
        main(*sys.argv[1:])
