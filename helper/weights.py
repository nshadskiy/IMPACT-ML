import array
from typing import Dict, Any, Union, List


def gen_weight(rdf: Any, sample_info: Dict[str, str]) -> Any:
    """
    Function to apply the generator weight and cross section.

    Args:
        rdf: root DataFrame object
        sample_info: Dictionary with information about a sample

    Return:
        root DataFrame object with the applied weight
    """
    number_generated_events_weight = 1.0 / float(sample_info["nevents"])
    cross_section_per_event_weight = float(sample_info["xsec"])
    negative_events_fraction = float(sample_info["generator_weight"])
    rdf = rdf.Define(
        "numberGeneratedEventsWeight", f"(float){number_generated_events_weight}"
    )
    rdf = rdf.Define(
        "crossSectionPerEventWeight", f"(float){cross_section_per_event_weight}"
    )
    rdf = rdf.Define("negativeEventsFraction", f"(float){negative_events_fraction}")

    return rdf.Redefine(
        "weight",
        "weight * numberGeneratedEventsWeight * crossSectionPerEventWeight * (( 1.0 / negativeEventsFraction) * ( ((genWeight<0) * -1) + ((genWeight>=0) * 1)))",
    )


def stitching_gen_weight(
    rdf: Any, era: str, process: str, sample_info: Dict[str, str]
) -> Any:
    """
    Function to apply the generator weight and cross section. This is specific for samples where stitching is used, like "DYjets" or "Wjets"

    Args:
        rdf: root DataFrame object
        era: Stitching weights depend on the data-taking period
        process: Stitching weights depend on the process e.g. "DYjets" or "Wjets"
        sample_info: Dictionary with information about a sample

    Return:
        root DataFrame object with the applied weight
    """
    number_generated_events_weight = 1.0 / float(sample_info["nevents"])
    cross_section_per_event_weight = float(sample_info["xsec"])
    negative_events_fraction = float(sample_info["generator_weight"])
    rdf = rdf.Define(
        "numberGeneratedEventsWeight", f"(float){number_generated_events_weight}"
    )
    rdf = rdf.Define(
        "crossSectionPerEventWeight", f"(float){cross_section_per_event_weight}"
    )
    rdf = rdf.Define("negativeEventsFraction", f"(float){negative_events_fraction}")

    if era == "2018":
        if process == "Wjets":
            rdf = rdf.Redefine(
                "weight",
                "weight * (0.0007590865*( ((npartons<=0) || (npartons>=5))*1.0 + (npartons==1)*0.2191273680 + (npartons==2)*0.1335837379 + (npartons==3)*0.0636217909 + (npartons==4)*0.0823135765 ))",
            )
        elif process == "DYjets":
            rdf = rdf.Redefine(
                "weight",
                "weight * ( (genbosonmass>=50.0)*0.0000631493*( ((npartons<=0) || (npartons>=5))*1.0 + (npartons==1)*0.2056921342 + (npartons==2)*0.1664121306 + (npartons==3)*0.0891121485 + (npartons==4)*0.0843396952 ) + (genbosonmass<50.0) * numberGeneratedEventsWeight * crossSectionPerEventWeight * (( 1.0 / negativeEventsFraction) * ( ((genWeight<0) * -1) + ((genWeight>=0) * 1))))",
            )
        else:
            raise ValueError(f"No stitching weights for this process: {process}")
    else:
        raise ValueError(f"No stitching weights defined for this era: {era}")

    return rdf


def lumi_weight(rdf: Any, era: str) -> Any:
    """
    Function to apply the luminosity depending on the era.

    Args:
        rdf: root DataFrame object
        era: Luminosity is depended on the data-taking period

    Return:
        root DataFrame object with the applied weight
    """
    if era == "2016preVFP":
        rdf = rdf.Redefine("weight", "weight * 19.52 * 1000.")
    elif era == "2016postVFP":
        rdf = rdf.Redefine("weight", "weight * 16.81 * 1000.")
    elif era == "2017":
        rdf = rdf.Redefine("weight", "weight * 41.48 * 1000.")
    elif era == "2018":
        rdf = rdf.Redefine("weight", "weight * 59.83 * 1000.")
    else:
        raise ValueError(f"Weight calc: lumi: Era is not defined: {era}")

    return rdf


def apply_btag_weight(rdf: Any) -> Any:
    """
    This function takes a b-tagger weight from the ntuples and calculates the yield correction factor
    for this weight dependent on the number of jets.
    The procedure is based on https://twiki.cern.ch/twiki/bin/view/CMS/BTagShapeCalibration#Effect_on_event_yields

    Args:
        rdf: root DataFrame object
    Return:
        root DataFrame with applied and corrected b-tagger weight
    """
    rdf = rdf.Define("wgt_with_btag", "weight * btag_weight")

    # measure corr. ratio for N jets (0 to 8); the highest N jet here is an arbitrary choice
    xbinning = array.array("d", [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
    nbinsx = len(xbinning) - 1

    # histogram without b-tagging SFs
    h = rdf.Histo1D(("", "", nbinsx, xbinning), "njets", "weight")
    h = h.GetValue()
    # histogram with b-tagging SFs
    h_btag = rdf.Histo1D(("", "", nbinsx, xbinning), "njets", "wgt_with_btag")
    h_btag = h_btag.GetValue()

    ratio = h.Clone()
    ratio.Divide(h_btag)

    # generating expression for b-tagging weights with the N jets dependent corr. ratio
    btag_wgt = "btag_weight*("
    for n in range(nbinsx):
        if ratio.GetBinContent(n + 1) != 0.0:
            btag_wgt += "(njets=={})*{}+".format(n, ratio.GetBinContent(n + 1))
        else:
            btag_wgt += "(njets=={})*{}+".format(n, 1.0)
    btag_wgt += "(njets>8)*1.)"

    # applying the b-tagging SFs
    rdf = rdf.Redefine("weight", "weight*{}".format(btag_wgt))

    return rdf


def apply_boostedtau_btag_weight(rdf: Any) -> Any:
    """
    This function takes a b-tagger weight from the ntuples and calculates the yield correction factor
    for this weight dependent on the number of jets.
    The procedure is based on https://twiki.cern.ch/twiki/bin/view/CMS/BTagShapeCalibration#Effect_on_event_yields

    Args:
        rdf: root DataFrame object
    Return:
        root DataFrame with applied and corrected b-tagger weight
    """
    rdf = rdf.Define("wgt_with_btag", "weight * btag_weight_boosted")

    # measure corr. ratio for N jets (0 to 8); the highest N jet here is an arbitrary choice
    xbinning = array.array("d", [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
    nbinsx = len(xbinning) - 1

    # histogram without b-tagging SFs
    h = rdf.Histo1D(("", "", nbinsx, xbinning), "njets_boosted", "weight")
    h = h.GetValue()
    # histogram with b-tagging SFs
    h_btag = rdf.Histo1D(("", "", nbinsx, xbinning), "njets_boosted", "wgt_with_btag")
    h_btag = h_btag.GetValue()

    ratio = h.Clone()
    ratio.Divide(h_btag)

    # generating expression for b-tagging weights with the N jets dependent corr. ratio
    btag_wgt = "btag_weight_boosted*("
    for n in range(nbinsx):
        if ratio.GetBinContent(n + 1) != 0.0:
            btag_wgt += "(njets_boosted=={})*{}+".format(n, ratio.GetBinContent(n + 1))
        else:
            btag_wgt += "(njets_boosted=={})*{}+".format(n, 1.0)
    btag_wgt += "(njets_boosted>8)*1.)"

    # applying the b-tagging SFs
    rdf = rdf.Redefine("weight", "weight*{}".format(btag_wgt))

    return rdf


def apply_tau_id_vsJet_weight(
    rdf: Any,
    channel: str,
    wp: str,
) -> Any:
    """
    This function applies tau id vs jet scale factors based on the working point which are chosen in the cuts.

    Args:
        rdf: root DataFrame object
        channel: Analysis channel of the tau analysis e.g. "et", "mt" or "tt"
        wp: A string defining the working point

    Return:
        root DataFrame with applied tau id vs jet scale factors
    """
    if channel in ["et", "mt"]:
        rdf = rdf.Redefine(
            "weight",
            f"weight * ((gen_match_2==5) * ((id_tau_vsJet_{wp}_2>0.5)*id_wgt_tau_vsJet_{wp}_2 + (id_tau_vsJet_{wp}_2<0.5)) + (gen_match_2!=5))",
        )

    elif channel == "tt":
        rdf = rdf.Redefine(
            "weight",
            f"weight * ((gen_match_1==5) * ((id_tau_vsJet_{wp}_1>0.5)*id_wgt_tau_vsJet_{wp}_1 + (id_tau_vsJet_{wp}_1<0.5)) + (gen_match_1!=5))",
        )
        rdf = rdf.Redefine(
            "weight",
            f"weight * ((gen_match_2==5) * ((id_tau_vsJet_{wp}_2>0.5)*id_wgt_tau_vsJet_{wp}_2 + (id_tau_vsJet_{wp}_2<0.5)) + (gen_match_2!=5))",
        )

    else:
        raise ValueError(
            f"Weight calc: tau id vs jet: Such a channel is not defined: {channel}"
        )

    return rdf


def apply_boostedtau_id_iso_weight(
    rdf: Any, channel: str, wp: Union[List[str], str]
) -> Any:
    """
    This function applies boosted tau iso id scale factors based on the working point which are chosen in the cuts.

    Args:
        rdf: root DataFrame object
        channel: Analysis channel of the tau analysis e.g. "et", "mt" or "tt"
        wp: A string including the working point

    Return:
        root DataFrame with applied boosted tau iso id scale factors
    """
    if channel in ["et", "mt"]:
        rdf = rdf.Redefine(
            "weight",
            f"weight * ((boosted_gen_match_2==5) * ((id_boostedtau_iso_{wp}_2>0.5)*id_wgt_boostedtau_iso_{wp}_2 + (id_boostedtau_iso_{wp}_2<0.5)) + (boosted_gen_match_2!=5))",
        )

    elif channel == "tt":
        rdf = rdf.Redefine(
            "weight",
            f"weight * ((boosted_gen_match_1==5) * ((id_boostedtau_iso_{wp}_1>0.5)*id_wgt_boostedtau_iso_{wp}_1 + (id_boostedtau_iso_{wp}_1<0.5)) + (boosted_gen_match_1!=5))",
        )
        rdf = rdf.Redefine(
            "weight",
            f"weight * ((boosted_gen_match_2==5) * ((id_boostedtau_iso_{wp}_2>0.5)*id_wgt_boostedtau_iso_{wp}_2 + (id_boostedtau_iso_{wp}_2<0.5)) + (boosted_gen_match_2!=5))",
        )

    else:
        raise ValueError(
            f"Weight calc: boosted tau id iso: Such a channel is not defined: {channel}"
        )

    return rdf


def apply_fake_factors(rdf: Any, channel: str, wp: str = None) -> Any:
    """
    This function applies fake factors. The have to be already calculated and are normally inside of friend tree files.

    Args:
        rdf: root DataFrame object
        channel: Analysis channel of the tau analysis e.g. "et", "mt" or "tt"
        wp: A string defining the working point, only relevant for "tt" channel

    Return:
        root DataFrame with applied fake factors
    """
    if channel in ["et", "mt"]:
        rdf = rdf.Redefine("weight", "weight * fake_factor")
    elif channel == "tt":
        rdf = rdf.Redefine(
            "weight",
            f"weight * (0.5 * fake_factor_1 * (id_tau_vsJet_{wp}_1 < 0.5) + 0.5 * fake_factor_2 * (id_tau_vsJet_{wp}_2 < 0.5))",
        )
    else:
        raise ValueError(
            f"Weight calc: fake factors: Such a channel is not defined: {channel}"
        )

    return rdf


def apply_boosted_fake_factors(rdf: Any, channel: str, wp: str = None) -> Any:
    """
    This function applies fake factors for boosted taus. The have to be already calculated and are normally inside of friend tree files.

    Args:
        rdf: root DataFrame object
        channel: Analysis channel of the boosted tau analysis e.g. "et", "mt" or "tt"
        wp: A string defining the working point, only relevant for "tt" channel

    Return:
        root DataFrame with applied boosted fake factors
    """
    if channel in ["et", "mt"]:
        rdf = rdf.Redefine("weight", "weight * fake_factor_boosted")
    elif channel == "tt":
        rdf = rdf.Redefine(
            "weight",
            f"weight * (0.5 * fake_factor_1_boosted * (id_boostedtau_iso_{wp}_1 < 0.5) + 0.5 * fake_factor_2_boosted * (id_boostedtau_iso_{wp}_2 < 0.5))",
        )
    else:
        raise ValueError(
            f"Weight calc: boosted fake factors: Such a channel is not defined: {channel}"
        )

    return rdf
