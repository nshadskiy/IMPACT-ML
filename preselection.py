"""
Script for preprocessing n-tuples for the neural network training
"""

import os
import argparse
import yaml
import json
import multiprocessing

from io import StringIO
from wurlitzer import pipes, STDOUT
import logging
from typing import Tuple, Dict, Union, List
import ROOT

import helper.filters as filters
import helper.weights as weights
import helper.functions as func


parser = argparse.ArgumentParser()

parser.add_argument(
    "--config-file",
    default=None,
    help="Path to the config file which contains information for the preselection step.",
)
parser.add_argument(
    "--nthreads",
    default=8,
    help="Number of threads to use for the preselection step. (default: 8)",
)
parser.add_argument(
    "--ncores",
    default=4,
    help="Number of cores to use for each pool the preselection step. (default: 4)",
)


def run_preselection(args: Tuple[str, Dict[str, Union[Dict, List, str]]]) -> None:
    """
    This function can be used for multiprocessing. It runs the preselection step for a specified process.

    Args:
        args: Tuple with a process name and a configuration for this process

    Return:
        None
    """
    process, config, ncores = args
    log = logging.getLogger(f"preselection.{process}")
    # ROOT.EnableImplicitMT(ncores) # multi-thread is not supported for rdf.Range()

    log.info(f"Processing process: {process}")
    # bookkeeping of samples files due to splitting based on the tau origin (genuine, jet fake, lepton fake)
    process_file_dict = dict()
    for tau_gen_mode in config["processes"][process]["tau_gen_modes"]:
        process_file_dict[tau_gen_mode] = list()
    log.info(
        f"Considered samples for process {process}: {config['processes'][process]['samples']}"
    )
    # define output variables
    outputs = config["output_feature"]
    if process in ["XToYHTo2Tau2B", "XToYHTo2B2Tau"]:
        outputs.append("gen_b_deltaR")

    # going through all contributing samples for the process
    for idx, sample in enumerate(config["processes"][process]["samples"]):
        # loading ntuple files
        ntuple_list = func.get_ntuples(config, process, sample)
        chain = ROOT.TChain(config["tree"])

        for ntuple in ntuple_list:
            chain.Add(ntuple)

        if "friends" in config:
            for friend in config["friends"]:
                if "fake_factors" not in friend or process == "tau_fakes":
                    friend_list = []
                    for ntuple in ntuple_list:
                        friend_list.append(
                            ntuple.replace("CROWNRun", "CROWNFriends/" + friend)
                        )
                    fchain = ROOT.TChain(config["tree"])
                    for friend in friend_list:
                        fchain.Add(friend)
                    chain.AddFriend(fchain)

        rdf = ROOT.RDataFrame(chain)

        if func.rdf_is_empty(rdf=rdf):
            log.info(f"WARNING: Sample {sample} is empty. Skipping...")
            continue

        rdf = func.add_mass_variables(rdf=rdf, sample=sample)

        if config["event_split"] == "even":
            rdf = rdf.Filter("(event%2==0)", "cut on even event IDs")
        elif config["event_split"] == "odd":
            rdf = rdf.Filter("(event%2==1)", "cut on odd event IDs")
        else:
            pass

        # apply general analysis event filters
        selection_conf = config["general_event_selection"]
        for cut in selection_conf:
            rdf = rdf.Filter(f"({selection_conf[cut]})", f"cut on {cut}")

        if process == "embedding":
            rdf = filters.emb_tau_gen_match(rdf=rdf, channel=config["channel"])

        # calculate event weights
        rdf = rdf.Define("weight", "1.")

        mc_weight_conf = config["mc_weights"]
        if process not in ["tau_fakes", "embedding"]:
            for weight in mc_weight_conf:
                if weight == "generator":
                    # calculating generator weight (including cross section weight)
                    if "stitching" in mc_weight_conf["generator"]: 
                        if process in mc_weight_conf["generator"]["stitching"]:
                            rdf = weights.stitching_gen_weight(
                                rdf=rdf,
                                era=config["era"],
                                process=process,
                                sample_info=datasets[sample],
                            )
                        else:
                            rdf = weights.gen_weight(rdf=rdf, sample_info=datasets[sample])
                    else:
                        rdf = weights.gen_weight(rdf=rdf, sample_info=datasets[sample])
                elif weight == "lumi":
                    rdf = weights.lumi_weight(rdf=rdf, era=config["era"])
                elif weight == "Z_pt_reweighting":
                    if process in ["DYjets"]:
                        rdf = rdf.Redefine(
                            "weight", f"weight * ({mc_weight_conf[weight]})"
                        )
                elif weight == "Top_pt_reweighting":
                    if process == "ttbar":
                        rdf = rdf.Redefine(
                            "weight", f"weight * ({mc_weight_conf[weight]})"
                        )
                else:
                    rdf = rdf.Redefine("weight", f"weight * ({mc_weight_conf[weight]})")

        if process == "embedding":
            emb_weight_conf = config["emb_weights"]
            for weight in emb_weight_conf:
                rdf = rdf.Redefine("weight", f"weight * ({emb_weight_conf[weight]})")

        # apply special analysis event filters: tau vs jet ID, btag
        selection_conf = config["special_event_selection"]
        for cut in selection_conf:
            if process == "tau_fakes":
                if "had_tau_id_vs_jet" in cut:
                    rdf = rdf.Filter(
                        f"({selection_conf['had_tau_id_vs_jet'][1]})",
                        "cut on had_tau_id_vs_jet",
                    )
                    # wp = (
                    #     selection_conf["had_tau_id_vs_jet"][1]
                    #     .rsplit("&&")[1]
                    #     .rsplit("_")[3]
                    # )
                    # rdf = weights.apply_fake_factors(
                    #     rdf=rdf, channel=config["channel"], wp=wp
                    # )
            else:
                if "had_tau_id_vs_jet" in cut:
                    rdf = rdf.Filter(
                        f"({selection_conf['had_tau_id_vs_jet'][0]})",
                        "cut on had_tau_id_vs_jet",
                    )
                    wp = selection_conf["had_tau_id_vs_jet"][0].rsplit("_")[3]
                    rdf = weights.apply_tau_id_vsJet_weight(
                        rdf=rdf, channel=config["channel"], wp=wp
                    )

            if "good_bb_pair" in cut:                    
                if process not in ["tau_fakes", "embedding"]:
                    rdf = weights.apply_pNet_weight(rdf=rdf)
                    rdf = weights.apply_btag_weight(rdf=rdf)
                rdf = rdf.Filter(
                    f"({selection_conf['good_bb_pair']})", "cut on good_bb_pair"
                )
                
            
        # additional variable definitions
        # rdf = rdf.Define("deltaPhi_met_tau1", "ROOT::VecOps::DeltaPhi(metphi, phi_1)*1")
        # rdf = rdf.Define("deltaPhi_met_tau2", "ROOT::VecOps::DeltaPhi(metphi, phi_2)*1")
        # rdf = rdf.Redefine("fj_Xbb_pt", "(abs(fj_Xbb_eta)<=2.5) ? fj_Xbb_pt : -10.;")
        # rdf = rdf.Redefine("fj_Xbb_phi", "(abs(fj_Xbb_eta)<=2.5) ? fj_Xbb_phi : -10.;")
        # rdf = rdf.Redefine("fj_Xbb_msoftdrop", "(abs(fj_Xbb_eta)<=2.5) ? fj_Xbb_msoftdrop : -10.;")
        # rdf = rdf.Redefine("fj_Xbb_nsubjettiness_2over1", "(abs(fj_Xbb_eta)<=2.5) ? fj_Xbb_nsubjettiness_2over1 : -10.;")
        # rdf = rdf.Redefine("fj_Xbb_nsubjettiness_3over2", "(abs(fj_Xbb_eta)<=2.5) ? fj_Xbb_nsubjettiness_3over2 : -10.;")
        # rdf = rdf.Redefine("fj_Xbb_mass", "(abs(fj_Xbb_eta)<=2.5) ? fj_Xbb_mass : -10.;")
        # rdf = rdf.Redefine("fj_Xbb_eta", "(abs(fj_Xbb_eta)<=2.5) ? fj_Xbb_eta : -10.;")

        # splitting data frame based on the tau origin (genuine, jet fake, lepton fake)
        for tau_gen_mode in config["processes"][process]["tau_gen_modes"]:
            tmp_rdf = rdf
            if tau_gen_mode != "all":
                tmp_rdf = filters.tau_origin_split(
                    rdf=tmp_rdf, channel=config["channel"], tau_gen_mode=tau_gen_mode
                )
            
            # introducing some preliminary balancing by reducing event numbers in larger samples
            if process in ["XToYHTo2Tau2B", "XToYHTo2B2Tau"]:
                if tmp_rdf.Count().GetValue() > 2000:
                    tmp_rdf = tmp_rdf.Range(0, 2000, 1)
            else:
                if tmp_rdf.Count().GetValue() > 100000:
                    tmp_rdf = tmp_rdf.Range(0, 100000, 1)

            # redirecting C++ stdout for Report() to python stdout
            out = StringIO()
            with pipes(stdout=out, stderr=STDOUT):
                tmp_rdf.Report().Print()
            log.info(out.getvalue())
            log.info("-" * 50)

            tmp_file_name = func.get_output_name(
                path=output_path, process=process, tau_gen_mode=tau_gen_mode, idx=idx
            )
            # check for empty data frame -> only save/calculate if event number is not zero
            if tmp_rdf.Count().GetValue() != 0:
                log.info(f"The current data frame will be saved to {tmp_file_name}")
                cols = tmp_rdf.GetColumnNames()
                cols_with_friends = [str(x).replace("ntuple.", "") for x in cols]
                missing_cols = [x for x in outputs if x not in cols_with_friends]
                if len(missing_cols) != 0:
                    raise ValueError(f"Missing columns: {missing_cols}")
                tmp_rdf.Snapshot(config["tree"], tmp_file_name, outputs)
                log.info("-" * 50)
                process_file_dict[tau_gen_mode].append(tmp_file_name)
            else:
                log.info("No events left after filters. Data frame will not be saved.")
                log.info("-" * 50)

    # combining all files of a process and tau origin
    for tau_gen_mode in config["processes"][process]["tau_gen_modes"]:
        out_file_name = func.get_output_name(
            path=output_path, process=process, tau_gen_mode=tau_gen_mode
        )
        # combining sample files to a single process file, if there are any
        if len(process_file_dict[tau_gen_mode]) != 0:
            sum_rdf = ROOT.RDataFrame(config["tree"], process_file_dict[tau_gen_mode])
            log.info(
                f"The processed files for the {process} process are concatenated. The data frame will be saved to {out_file_name}"
            )
            sum_rdf.Snapshot(config["tree"], out_file_name, outputs)
            log.info("-" * 50)
        else:
            log.info(
                f"No processed files for the {process} process. An empty data frame will be saved to {out_file_name}"
            )
            # create an empty root file and save it
            f = ROOT.TFile(out_file_name, "RECREATE")
            t = ROOT.TTree(config["tree"], config["tree"])
            t.Write()
            f.Close()
            log.info("-" * 50)

        # delete not needed temporary sample files after combination
        for rf in process_file_dict[tau_gen_mode]:
            os.remove(rf)
        log.info("-" * 50)


if __name__ == "__main__":
    args = parser.parse_args()

    # loading of the chosen config file
    with open(args.config_file, "r") as file:
        config = yaml.load(file, yaml.FullLoader)

    # loading general dataset info file for xsec and event number
    with open("datasets/datasets.json", "r") as file:
        datasets = json.load(file)

    # define output path for the preselected samples
    output_path = os.path.join(
        config["output_path"],
        "preselection",
        config["era"],
        config["channel"],
        config["event_split"],
    )
    func.check_path(path=output_path)

    func.setup_logger(
        log_file=output_path + "/preselection.log",
        log_name="preselection",
        subcategories=config["processes"],
    )

    mass_combinations = func.get_mass_combinations(
        masses_X=config["masses_X"], masses_Y=config["masses_Y"]
    )
    
    if "XToYHTo2Tau2B" in config["processes"]: 
        YttHbb_samples = list()
        for comb in mass_combinations:
            YttHbb_string = config["processes"]["XToYHTo2Tau2B"]["samples"][0]
            YttHbb_samples.append(
                YttHbb_string.replace("massX", comb["massX"]).replace(
                    "massY", comb["massY"]
                )
            )
        config["processes"]["XToYHTo2Tau2B"]["samples"] = YttHbb_samples
        
    if "XToYHTo2B2Tau" in config["processes"]:
        YbbHtt_samples = list()
        for comb in mass_combinations:
            YbbHtt_string = config["processes"]["XToYHTo2B2Tau"]["samples"][0]
            YbbHtt_samples.append(
                YbbHtt_string.replace("massX", comb["massX"]).replace(
                    "massY", comb["massY"]
                )
            )
        config["processes"]["XToYHTo2B2Tau"]["samples"] = YbbHtt_samples

    # going through all wanted processes and run the preselection function with a pool of 8 workers
    args_list = [(process, config, int(args.ncores)) for process in config["processes"]]

    with multiprocessing.Pool(
        processes=min(len(config["processes"]), int(args.nthreads))
    ) as pool:
        pool.map(run_preselection, args_list)

    # dumping mass grid to output directory for documentation
    with open(output_path + "/mass_combinations.yaml", "w") as config_file:
        yaml.dump(mass_combinations, config_file, default_flow_style=False)

    # dumping config to output directory for documentation
    with open(output_path + "/config.yaml", "w") as config_file:
        yaml.dump(config, config_file, default_flow_style=False)
