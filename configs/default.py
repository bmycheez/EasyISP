from mmengine.config import Config

values = {
    "main": {
        "constant": {
            "name": "EasyISP",
            "icon_path": "assets/logo_enerzai.ico",
            "ext": {
                "raw": [["Numpy Files", "*.npy"],
                        ["Raw Files", "*.raw"]],
                "rgb": [["PNG Files", "*.png"],
                        ["JPEG Files", "*.jpg"]]
                    },
            "folder_len": 70,
            "file_len": 40,
            "domain_len": 39,
            "text_len": 5,
            "mini_len": 2,
            "scroll_column_size": [300, 400]
        },
        "bit_depth": 8,
        "width": 1920, "height": 1080,
        "config": "configs/main.yaml",
        "in_path": ".",
        "out_path": ".",
        "cpu": False, "cuda": True,
        "gbrg": False, "bggr": False, "rggb": False, "grbg": True,
        "srgb": True,
    },
    "easy": {
        "blc": True, "blc_bl": 256,
        "cfa": True,
        "awb": True, "awb_gw": True, "awb_gain": {"r": 2, "g": 1, "b": 2},
        "ccm": True, "ccm_matrix":
        {"r2r": 1.8, "r2g": -0.8, "r2b": 0,
         "g2r": -0.3, "g2g": 1.5, "g2b": -0.2,
         "b2r": 0, "b2g": -0.8, "b2b": 1.8},
        "gac": True, "gac_gain": 0.1, "gac_gamma": 2.2
    },
    "fast": {
        "dpc": False, "dpc_diff": 30,
        "blc": True, "blc_bl": 256,
        "lsc": True, "lsc_intensity": 0,
        "aaf": True,
        "awb": True, "awb_gw": True, "awb_gain":
        {"r": 1860, "gr": 1024, "gb": 1024, "b": 1280},
        "cnf": False, "cnf_diff": 0,
        "cfa": True, "cfa_malvar": True, "cfa_bilinear": False,
        "ccm": True, "ccm_matrix":
        {"r2r": 1024, "r2g": 0, "r2b": 0, "r2o": 0,
         "g2r": 0, "g2g": 1024, "g2b": 0, "g2o": 0,
         "b2r": 0, "b2g": 0, "b2b": 1024, "b2o": 0},
        "gac": True, "gac_gain": 256, "gac_gamma": 0.42,
        "csc": True,
        "nlm": False,
        "nlm_search_window_size": 9, "nlm_patch_size": 3, "nlm_h": 10,
        "bnf": False, "bnf_intensity_sigma": 0.8, "bnf_spatial_sigma": 0.8,
        "ceh": True,
        "ceh_tile_width": 4, "ceh_tile_height": 6, "ceh_clip_limit": 0.01,
        "eeh": True, "eeh_gain": 384, "eeh_flat_threshold": 4,
        "eeh_edge_threshold": 8, "eeh_delta_threshold": 64,
        "fcs": True, "fcs_delta_min": 8, "fcs_delta_max": 32,
        "hsc": True, "hsc_offset": 0, "hsc_gain": 256,
        "bcc": True, "bcc_offset": 0, "bcc_gain": 256,
        "scl": False,
        "stop": ["dpc", "blc", "aaf", "awb", "cnf", "cfa", "ccm", "gac"],
    },
    "nr3d": {
        "fps": 30,
        "nr3d": False, "filter": False,
        "filter_nr3d": False, "nr3d_filter": True,
        "num_frames": 2, "threshold": 0.2,
        "eeh": True,
        "eeh_ksize": 3, "eeh_edge_factor": 0.3, "eeh_signal_factor": 1,
        "lpf_luma": True, "lpf_luma_bnf": True, "lpf_luma_median": False,
        "lpf_luma_ksize": 3, "lpf_luma_bnf_sigma": {"color": 75, "space": 75},
        "lpf_chroma": True, "lpf_chroma_median": True, "lpf_chroma_ksize": 9,
        "ffmpeg":
        "ffmpeg -i {0} -c:v libx264 -crf 8 -y -hide_banner -loglevel error {1}"
    },

}

main = Config(values).main
easy = Config(values).easy
fast = Config(values).fast
nr3d = Config(values).nr3d
