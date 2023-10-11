import os
import logging
import yaml
from typing import List, Dict, Union

import pandas as pd
import awkward as ak
import numpy as np
import uproot

from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.utils import shuffle

log = logging.getLogger("training")


class Data:
    def __init__(
        self, feature_list: List[str], class_dict: Dict[str, List], signal: List[str]
    ):
        """
        Initialize a data object by defining features and classes. To load data or transform it, dedicated methods are implemented.

        Args:
                feature_list: List of strings with feature names e.g. ["bpair_eta_1","m_vis"]
                class_dict: Dict with the classes as keys and a list of coresponding files/processes as values e.g. {"misc": ["DYjets_L","diboson_L"]}
                signal: List with the signal classes
        """
        self.features = feature_list
        self.param_features = list()

        if "massX" in self.features:
            self.features.remove("massX")
            self.param_features.append("massX")
        if "massY" in self.features:
            self.features.remove("massY")
            self.param_features.append("massY")

        self.classes = class_dict.keys()
        self.file_dict = class_dict
        self.signal = signal

    def load_data(self, sample_path: str, era: str, channel: str) -> None:
        """
        General function to load data from root files into a pandas DataFrame.

        Args:
                sample_path: Absolute path to root files e.g. "/ceph/path/to/root_files"
                era: Data taking era e.g. "2018"
                channel: Analysis channel e.g. "mt" for the mu tau channel in a tau analysis

        Return:
                None
        """
        self.sample_path = sample_path
        self.era = era
        self.channel = channel
        self.label_dict = dict()

        with open(
            os.path.join(
                self.sample_path,
                "preselection",
                self.era,
                self.channel,
                "mass_combinations.yaml",
            ),
            "r",
        ) as file:
            self.mass_combinations = yaml.load(file, yaml.FullLoader)

        log.info("-" * 50)
        log.info("loading samples from " + str(self.sample_path))
        self.samples = self._load_samples()
        self.df_all = pd.concat(self.samples, sort=True)
        self._balance_samples()
        if len(self.mass_combinations) > 0:
            self._balance_signal_samples()
        self.df_all = self.df_all.reset_index(drop=True)

        del self.samples

    def transform(self, type: str, one_hot: bool) -> None:
        """
        Transforms the features to a standardized range.

        Args:
                type: Options are "standard" (shifts feature distributions to mu=0 and sigma=1) or "quantile" (transforms feature distributions into Normal distributions with mu=0 and sigma=1)
                one_hot: Boolean to decide how to include the parametrization variables. True: as one hot encoded features, False: as single integer features with values from 0 in steps of 1

        Return:
                None
        """
        self.transform_type = type

        self.mass_indizes = {"massX": dict(), "massY": dict()}
        if len(self.param_features) > 0:
            nX, nY = 0, 0
            for comb in self.mass_combinations:
                if comb["massX"] not in self.mass_indizes["massX"]:
                    self.mass_indizes["massX"][comb["massX"]] = nX
                    nX += 1
                if comb["massY"] not in self.mass_indizes["massY"]:
                    self.mass_indizes["massY"][comb["massY"]] = nY
                    nY += 1
            log.info("-" * 50)
            log.info(f"mass indizes: {self.mass_indizes}")

            if one_hot:
                tmp_param_features = list()
                for param in self.param_features:
                    trans_masses = [
                        self.mass_indizes[param][str(mass)]
                        for mass in self.df_all[param]
                    ]
                    n_diff_masses = np.max(trans_masses) + 1
                    trans_masses = np.eye(n_diff_masses)[trans_masses]
                    for n in range(n_diff_masses):
                        self.df_all[f"{param}_{n}"] = trans_masses[:, n]
                        tmp_param_features.append(f"{param}_{n}")

                self.param_features = tmp_param_features
                log.info(f"final parametrization variables: {self.param_features}")
            else:
                for param in self.param_features:
                    trans_masses = [
                        self.mass_indizes[param][str(mass)]
                        for mass in self.df_all[param]
                    ]
                    self.df_all[param] = trans_masses

        if self.transform_type == "standard":
            log.info("-" * 50)
            log.info("Standard Transformation")
            st = StandardScaler()
            st_df = pd.DataFrame(
                data=st.fit_transform(self.df_all[self.features]),
                columns=self.features,
                index=self.df_all.index,
            )
            for feature in self.features:
                self.df_all[feature] = st_df[feature]
            del st_df
            log.debug(st.mean_)
            log.debug(st.scale_)
            # self.df_all[self.features] = ( self.df_all[self.features] - self.df_all[self.features].mean() ) / self.df_all[self.features].std()

        elif self.transform_type == "quantile":
            log.info("-" * 50)
            log.info("Quantile Transformation")
            qt = QuantileTransformer(n_quantiles=500, output_distribution="normal")
            qt_df = pd.DataFrame(
                data=qt.fit_transform(self.df_all[self.features]),
                columns=self.features,
                index=self.df_all.index,
            )
            for feature in self.features:
                self.df_all[feature] = qt_df[feature]
            del qt_df

        else:
            raise ValueError("wrong transformation type!")

    def shuffling(self, seed: Union[int, None] = None):
        """
        Shuffles data events according to a shuffle seed.

        Args:
                seed: Integer which is used as shuffle seed (default is a random seed)

        Return:
                None
        """
        self.shuffle_seed = np.random.randint(low=0, high=2**16)
        if seed is not None:
            self.shuffle_seed = seed

        log.info("-" * 50)
        log.info(f"using {self.shuffle_seed} as seed to shuffle data")
        self.df_all = shuffle(self.df_all, random_state=self.shuffle_seed)

    def split_data(
        self, train_fraction: float = 0.7, val_fraction: float = 0.2
    ) -> None:
        """
        Splits the data into a training, validation and test set.

        Args:
                train_fraction: Float number as fraction of the data that should be used for training.
                val_fraction: Float number as fraction of the training data that should be used for validation.

        Return:
                None
        """
        # defining test dataset
        n_test = int(self.df_all.shape[0] * (1 - train_fraction))

        self.df_test = self.df_all.head(n_test)
        self.df_test_labels = self.df_test["label"]
        self.df_test_weights = self.df_test["plot_weight"]

        self.df_test = self.df_test[self.features + self.param_features]

        # defining train dataset
        df_train_val = self.df_all.tail(self.df_all.shape[0] - n_test)
        n_val = int(df_train_val.shape[0] * val_fraction)

        self.df_train = df_train_val.tail(df_train_val.shape[0] - n_val)
        self.df_train_labels = self.df_train["label"]
        self.df_train_weights = self.df_train["weight"]

        self.df_train = self.df_train[self.features + self.param_features].values
        self.df_train_labels = self.df_train_labels.values
        self.df_train_weights = self.df_train_weights.values

        self.df_val = df_train_val.head(n_val)
        self.df_val_labels = self.df_val["label"]
        self.df_val_weights = self.df_val["weight"]

        self.df_val = self.df_val[self.features + self.param_features].values
        self.df_val_labels = self.df_val_labels.values
        self.df_val_weights = self.df_val_weights.values

        del df_train_val

    #########################################################################################
    ### private functions ###
    #########################################################################################

    def _load_samples(self) -> List[pd.DataFrame]:
        """
        Loading data from root files into a pandas DataFrame based on defined classes for the neural network task.

        Args:
                None

        Return:
                List of pandas DataFrames with one DataFrame for each class
        """
        class_data = list()

        for cl in self.file_dict:
            log.info("-" * 50)
            log.info(f"loading {cl} class")

            tmp_file_dict = dict()
            # define a dictionary of all files in a class which is then used to load the data with uproot
            for file in self.file_dict[cl]:
                tmp_file_dict[
                    os.path.join(
                        self.sample_path,
                        "preselection",
                        self.era,
                        self.channel,
                        file + ".root",
                    )
                ] = "ntuple"
            log.info(tmp_file_dict)

            events = uproot.concatenate(tmp_file_dict)
            # transform the loaded awkward array to a pandas DataFrame
            df = ak.to_dataframe(events)
            df = self._randomize_masses(df, cl)
            df = self._add_labels(df, cl)
            log.info(f"number of events for {cl}: {df.shape[0]}")

            class_data.append(df.copy())

        return class_data

    def _randomize_masses(self, df: pd.DataFrame, cl: str) -> pd.DataFrame:
        """y_test
        Adding random mass points for backgrounds.

        Args:
                df: Input DataFrame for one class
                cl: Name of the class

        Return:
                Pandas DataFrames with random mass points
        """
        if cl not in self.signal:
            rand_masses = np.random.choice(self.mass_combinations, len(df))
            rand_masses = np.array(
                [[int(m["massX"]), int(m["massY"])] for m in rand_masses]
            )

            df["massX"] = rand_masses[:, 0]
            df["massY"] = rand_masses[:, 1]

        return df

    def _add_labels(self, df: pd.DataFrame, cl: str) -> pd.DataFrame:
        """
        Adding labels to a DataFrame for a classification task for one specified class.

        Args:
                df: Input DataFrame for one class
                cl: Name of the class to which labels should be added

        Return:
                Pandas DataFrames with added labels
        """
        if cl not in self.label_dict:
            # create an index encoded label based on the number of classes if the label wasn't defined yet
            self.label_dict[cl] = len(self.label_dict)
        else:
            pass
        log.debug(self.label_dict)

        # add a column with the label to the DateFrame of the class
        df["label"] = [self.label_dict[cl] for _ in range(df.shape[0])]

        return df

    def _balance_samples(self):
        """
        Normalize the event weights to balance different event numbers of classes.

        Args:
                None

        Return:
                None
        """
        log.info("-" * 50)
        self.df_all["plot_weight"] = self.df_all["weight"]
        sum_weights_all = sum(self.df_all["weight"].values)
        for cl in self.classes:
            mask = self.df_all["label"].isin([self.label_dict[cl]])
            log.info(
                f"weight sum before class balance for {cl}: {sum(self.df_all.loc[mask, 'weight'].values)}"
            )
            self.df_all.loc[mask, "weight"] = self.df_all.loc[mask, "weight"] * (
                sum_weights_all
                / (len(self.classes) * sum(self.df_all.loc[mask, "weight"].values))
            )
            log.info(
                f"weight sum after class balance for {cl}: {sum(self.df_all.loc[mask, 'weight'].values)}"
            )

    def _balance_signal_samples(self):
        """
        Normalize the signal event weights to balance different event numbers of different mass point combinations.

        Args:
                None

        Return:
                None
        """
        log.info("-" * 50)
        for sig in self.signal:
            df_sig = self.df_all[self.df_all["label"] == self.label_dict[sig]]
            sum_weights_sig = sum(df_sig["weight"].values)
            for comb in self.mass_combinations:
                mask = (
                    (self.df_all["label"] == self.label_dict[sig])
                    & (self.df_all["massX"] == int(comb["massX"]))
                    & (self.df_all["massY"] == int(comb["massY"]))
                )
                log.info(
                    f"event number massX {comb['massX']}, massY {comb['massY']}: {sum(mask)}"
                )
                log.info(
                    f"weight sum before signal mass balance for {sig}, massX {comb['massX']}, massY {comb['massY']}: {sum(self.df_all.loc[mask, 'weight'].values)}"
                )
                self.df_all.loc[mask, "weight"] = self.df_all.loc[mask, "weight"] * (
                    sum_weights_sig
                    / (
                        len(self.mass_combinations)
                        * sum(self.df_all.loc[mask, "weight"].values)
                    )
                )
                log.info(
                    f"weight sum after signal mass balance for {sig}, massX {comb['massX']}, massY {comb['massY']}: {sum(self.df_all.loc[mask, 'weight'].values)}"
                )
