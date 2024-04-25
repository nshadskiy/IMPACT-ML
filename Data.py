import os
import logging
import yaml
from typing import List, Dict, Union, Any

import pandas as pd
import awkward as ak
import numpy as np
import uproot

from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.utils import shuffle

log = logging.getLogger("training")


class Data:
    def __init__(
        self,
        feature_list: List[str],
        class_dict: Dict[str, List],
        config: Dict[str, Any],
        event_split: str,
    ):
        """
        Initialize a data object by defining features and classes. To load data or transform it, dedicated methods are implemented.

        Args:
                feature_list: List of strings with feature names e.g. ["bpair_eta_1","m_vis"]
                class_dict: Dict with the classes as keys and a list of coresponding files/processes as values e.g. {"misc": ["DYjets_L","diboson_L"]}
                config: Dictionary with information for data processing
                event_split: String to define if even or odd event IDs should be used for training. The other is used for testing.
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
        self.signal = config["signal"]
        self.config = config
        self.event_split = event_split

    def load_data(self, sample_path: str, era: str, channel: str, shuffle_seed: Union[int, None] = None, val_fraction: float = 0.2) -> None:
        """
        General function to load data from root files into a pandas DataFrame.

        Args:
                sample_path: Absolute path to root files e.g. "/ceph/path/to/root_files"
                era: Data taking era e.g. "2018"
                channel: Analysis channel e.g. "mt" for the mu tau channel in a tau analysis
                shuffle_seed: Integer which is used as shuffle seed (default is a random seed)
                val_fraction: Float number as fraction of the training data that should be used for validation

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
                self.event_split,
                "mass_combinations.yaml",
            ),
            "r",
        ) as file:
            self.mass_combinations = yaml.load(file, yaml.FullLoader)
        self.plot_mass_combinations = self.config["plot_mass_combinations"]

        log.info("-" * 50)
        log.info(f"loading {self.event_split} samples from {self.sample_path} for training")
        self.samples_train, self.samples_val = self._load_training_samples(event_ID=self.event_split, shuffle_seed=shuffle_seed, val_fraction=val_fraction)
        self.df_train = pd.concat(self.samples_train)
        self.df_val = pd.concat(self.samples_val)
        log.info("-" * 50)
        if self.event_split == "even":
            log.info(f"loading odd samples from {self.sample_path} for testing")
            self.samples_test = self._load_testing_samples(event_ID="odd")
        elif self.event_split == "odd":
            log.info(f"loading even samples from {self.sample_path} for testing")
            self.samples_test = self._load_testing_samples(event_ID="even")
        else:
            raise ValueError("Event split wrongly defined.")

        self.df_test = pd.concat(self.samples_test, sort=True)
        
        log.info("-" * 50)
        log.info("balancing classes in training dataset")
        self._balance_samples()
        
        if len(self.mass_combinations) > 0:
            log.info("-" * 50)
            log.info("balancing signal in training dataset")
            self._balance_signal_samples()
            
        # self.df_train = self.df_train.reset_index(drop=True)
        # self.df_val = self.df_val.reset_index(drop=True)
        self.df_test = self.df_test.reset_index(drop=True)

        del self.samples_train
        del self.samples_test

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
                        for mass in self.df_train[param]
                    ]
                    n_diff_masses = np.max(trans_masses) + 1
                    trans_masses = np.eye(n_diff_masses)[trans_masses]
                    for n in range(n_diff_masses):
                        self.df_train[f"{param}_{n}"] = trans_masses[:, n]
                        tmp_param_features.append(f"{param}_{n}")

                    trans_masses = [
                        self.mass_indizes[param][str(mass)]
                        for mass in self.df_val[param]
                    ]
                    n_diff_masses = np.max(trans_masses) + 1
                    trans_masses = np.eye(n_diff_masses)[trans_masses]
                    for n in range(n_diff_masses):
                        self.df_val[f"{param}_{n}"] = trans_masses[:, n]
                        
                    trans_masses = [
                        self.mass_indizes[param][str(mass)]
                        for mass in self.df_test[param]
                    ]
                    n_diff_masses = np.max(trans_masses) + 1
                    trans_masses = np.eye(n_diff_masses)[trans_masses]
                    for n in range(n_diff_masses):
                        self.df_test[f"{param}_{n}"] = trans_masses[:, n]

                self.param_features = tmp_param_features
                log.info(f"final parametrization variables: {self.param_features}")
            else:
                for param in self.param_features:
                    trans_masses = [
                        self.mass_indizes[param][str(mass)]
                        for mass in self.df_train[param]
                    ]
                    self.df_train[param] = trans_masses
                    
                    trans_masses = [
                        self.mass_indizes[param][str(mass)]
                        for mass in self.df_val[param]
                    ]
                    self.df_val[param] = trans_masses
                    
                    trans_masses = [
                        self.mass_indizes[param][str(mass)]
                        for mass in self.df_test[param]
                    ]
                    self.df_test[param] = trans_masses

        if self.transform_type == "standard":
            log.info("-" * 50)
            log.info("Standard Transformation")
            st = StandardScaler()
            st.fit(self.df_train[self.features])
            
            self.transform_feature_dict = dict()
            for idx, feature in enumerate(self.features):
                self.transform_feature_dict[feature] = {"mean": st.mean_[idx], "std": st.scale_[idx]}

            st_df_train = pd.DataFrame(
                data=st.transform(self.df_train[self.features]),
                columns=self.features,
                index=self.df_train.index,
            )
            for feature in self.features:
                self.df_train[feature] = st_df_train[feature]

            st_df_val = pd.DataFrame(
                data=st.transform(self.df_val[self.features]),
                columns=self.features,
                index=self.df_val.index,
            )
            for feature in self.features:
                self.df_val[feature] = st_df_val[feature]
                
            st_df_test = pd.DataFrame(
                data=st.transform(self.df_test[self.features]),
                columns=self.features,
                index=self.df_test.index,
            )
            for feature in self.features:
                self.df_test[feature] = st_df_test[feature]

            del st_df_train
            del st_df_val
            del st_df_test
            log.debug(st.mean_)
            log.debug(st.scale_)
            # self.df_all[self.features] = ( self.df_all[self.features] - self.df_all[self.features].mean() ) / self.df_all[self.features].std()

        elif self.transform_type == "quantile":
            log.info("-" * 50)
            log.info("Quantile Transformation")
            qt = QuantileTransformer(n_quantiles=500, output_distribution="normal")
            qt.fit(self.df_train[self.features])

            qt_df_train = pd.DataFrame(
                data=qt.transform(self.df_train[self.features]),
                columns=self.features,
                index=self.df_train.index,
            )
            for feature in self.features:
                self.df_train[feature] = qt_df_train[feature]

            qt_df_val = pd.DataFrame(
                data=qt.transform(self.df_val[self.features]),
                columns=self.features,
                index=self.df_val.index,
            )
            for feature in self.features:
                self.df_val[feature] = qt_df_val[feature]
                
            qt_df_test = pd.DataFrame(
                data=qt.transform(self.df_test[self.features]),
                columns=self.features,
                index=self.df_test.index,
            )
            for feature in self.features:
                self.df_test[feature] = qt_df_test[feature]

            del qt_df_train
            del qt_df_val
            del qt_df_test

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
        self.df_train = shuffle(self.df_train, random_state=self.shuffle_seed)
        self.df_val = shuffle(self.df_val, random_state=self.shuffle_seed)

    def prepare_for_training(self) -> None:
        """
        Prepare data for training by spliting training features, label and weights.

        Args:
                None

        Return:
                None
        """
        log.info("-" * 50)
        # defining test dataset
        self.df_test_labels = self.df_test["label"]
        self.df_test_weights = self.df_test["weight"]

        self.df_test = self.df_test[self.features + self.param_features]

        # defining train dataset
        self.df_train_labels = self.df_train["label"]
        self.df_train_weights = self.df_train["weight"]

        self.df_train = self.df_train[self.features + self.param_features].values
        self.df_train_labels = self.df_train_labels.values
        self.df_train_weights = self.df_train_weights.values

        # defining validation dataset
        self.df_val_labels = self.df_val["label"]
        self.df_val_weights = self.df_val["weight"]

        self.df_val = self.df_val[self.features + self.param_features].values
        self.df_val_labels = self.df_val_labels.values
        self.df_val_weights = self.df_val_weights.values


    #########################################################################################
    ### private functions ###
    #########################################################################################

    def _load_training_samples(self, event_ID: str, shuffle_seed: Union[int, None] = None, val_fraction: float = 0.2) -> List[pd.DataFrame]:
        """
        Loading data from root files into a pandas DataFrame based on defined classes for the neural network task.

        Args:
                event_ID: String to specify to select events with "even" or "odd" IDs
                shuffle_seed: Integer which is used as shuffle seed (default is a random seed)
                val_fraction: Float number as fraction of the training data that should be used for validation

        Return:
                List of pandas DataFrames with one DataFrame for each class
        """
        class_data_train = list()
        class_data_val = list()

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
                        event_ID,
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
            df = df.reset_index(drop=True)
            
            # shuffle data before training/validation, especially relevant for multi mass signal samples
            self.shuffle_seed = np.random.randint(low=0, high=2**16)
            if shuffle_seed is not None:
                self.shuffle_seed = shuffle_seed

            log.info("-" * 50)
            log.info(f"using {self.shuffle_seed} as seed to shuffle data")
            df = shuffle(df, random_state=self.shuffle_seed)
            
            # split samples in training and validation
            n_val = int(df.shape[0] * val_fraction)
            df_train = df.tail(df.shape[0] - n_val)
            df_val = df.head(n_val)

            class_data_train.append(df_train.copy())
            class_data_val.append(df_val.copy())

        return class_data_train, class_data_val
    
    
    def _load_testing_samples(self, event_ID: str) -> List[pd.DataFrame]:
        """
        Loading data from root files into a pandas DataFrame based on defined classes for the neural network task.

        Args:
                event_ID: String to specify to select events with "even" or "odd" IDs

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
                        event_ID,
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
        """
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

        if cl in self.signal:
            # split data into boosted bb and resolved bb
            if "_res" in cl:
                df = df[
                    (df["gen_b_deltaR"] >= self.config["bb_deltaR_split_value"])
                ].copy(deep=True)
            elif "_boost" in cl:
                df = df[
                    (df["gen_b_deltaR"] < self.config["bb_deltaR_split_value"])
                ].copy(deep=True)
            else:
                pass

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
        sum_weights_all = sum(self.df_train["weight"].values) + sum(self.df_val["weight"].values)
        for cl in self.classes:
            mask_train = self.df_train["label"].isin([self.label_dict[cl]])
            mask_val = self.df_val["label"].isin([self.label_dict[cl]])
            sum_weights_class = sum(self.df_train.loc[mask_train, 'weight'].values) + sum(self.df_val.loc[mask_val, 'weight'].values)
            log.info(
                f"weight sum before class balance for {cl}: {sum_weights_class}"
            )
            self.df_train.loc[mask_train, "weight"] = self.df_train.loc[mask_train, "weight"] * (sum_weights_all / (len(self.classes) * sum_weights_class))
            self.df_val.loc[mask_val, "weight"] = self.df_val.loc[mask_val, "weight"] * (sum_weights_all / (len(self.classes) * sum_weights_class))
            
            sum_weights_class_new = sum(self.df_train.loc[mask_train, 'weight'].values) + sum(self.df_val.loc[mask_val, 'weight'].values)
            log.info(
                f"weight sum after class balance for {cl}: {sum_weights_class_new}"
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
            df_sig_train = self.df_train[
                self.df_train["label"] == self.label_dict[sig]
            ]
            df_sig_val = self.df_val[
                self.df_val["label"] == self.label_dict[sig]
            ]
            sum_weights_sig = sum(df_sig_train["weight"].values) + sum(df_sig_val["weight"].values)
            for comb in self.mass_combinations:
                mask_train = (
                    (self.df_train["label"] == self.label_dict[sig])
                    & (self.df_train["massX"] == int(comb["massX"]))
                    & (self.df_train["massY"] == int(comb["massY"]))
                )
                mask_val = (
                    (self.df_val["label"] == self.label_dict[sig])
                    & (self.df_val["massX"] == int(comb["massX"]))
                    & (self.df_val["massY"] == int(comb["massY"]))
                )
                evt_num = sum(mask_train)+sum(mask_val)
                log.info(
                    f"event number massX {comb['massX']}, massY {comb['massY']}: {evt_num}"
                )
                sum_weights_sig_mass = sum(self.df_train.loc[mask_train, 'weight'].values) + sum(self.df_val.loc[mask_val, 'weight'].values)
                log.info(
                    f"weight sum before signal mass balance for {sig}, massX {comb['massX']}, massY {comb['massY']}: {sum_weights_sig_mass}"
                )
                if evt_num > 0:
                    if sum(mask_train) > 0:
                        self.df_train.loc[mask_train, "weight"] = self.df_train.loc[mask_train, "weight"] * (sum_weights_sig / (len(self.mass_combinations) * sum_weights_sig_mass))
                    if sum(mask_val) > 0:
                        self.df_val.loc[mask_val, "weight"] = self.df_val.loc[mask_val, "weight"] * (sum_weights_sig / (len(self.mass_combinations) * sum_weights_sig_mass))
                    
                    sum_weights_all_new = sum(self.df_train.loc[mask_train, 'weight'].values) + sum(self.df_val.loc[mask_val, 'weight'].values)
                    log.info(
                        f"weight sum after signal mass balance for {sig}, massX {comb['massX']}, massY {comb['massY']}: {sum_weights_all_new}"
                    )
                else:
                    log.info(
                        f"no signal mass balance for {sig}, massX {comb['massX']}, massY {comb['massY']} because no events selected"
                    )
