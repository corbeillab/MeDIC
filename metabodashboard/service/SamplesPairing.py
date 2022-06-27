from sklearn.model_selection import train_test_split
import glob, random
import numpy as np

class SamplesPairing():
    def __init__(self, pairings, sample_names, targets, IDs, proportion_in_test, nbr_splits):
        self.pairings = pairings
        self.sample_names = sample_names
        self.targets = targets
        self.ids = IDs
        self.proportion = float(proportion_in_test)
        self.nbr_splits = int(nbr_splits)
        self.dict_splits = {}

        self.names_dict = {n: idx for idx, n in enumerate(self.sample_names)}


    def split(self):
        """
        We assume there can be maximum two types of pairing simultaneously
        :return: Nothing
        """
        if not self._is_there_pairing_to_do():

            ###### Creation de X et y ######

            X = []
            y = []
            for s in self.sample_names:  # itère sur chq nom de sample
                for i, id in enumerate(self.ids):  # itère sur chq id unique
                    if id in s:  # vérifie si l'id est présent dans le nom de sample
                        X.append(s)
                        y.append(self.targets[i])  # ajoute label correspondant à l'id(correspondant lui-meme au sample)

            ##### Create the splits ######

            for i in range(self.nbr_splits):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.proportion,
                                                                    random_state=i)
                # convert sample names to indices so its easier to reconstruct later
                # X_train = [self.names_dict[i] for i in X_train]
                # X_test = [self.names_dict[i] for i in X_test]

                ###### save splits dans dict ######

                self.dict_splits["split{}".format(i)] = [X_train, X_test, y_train, y_test]
        else:
            self._test_pairing_patterns_spelling()
            # select base files to do the spliting on

            ###### Creation de X et y ######

            X = []
            y = []
            first_patterns = [i[0] for i in self.pairings]
            for s in self.sample_names:  # itère sur chaque nom de sample
                ok = True
                for p in first_patterns:  # itère sur chq 1er pattern des pairings
                    if p not in s:  # check si le (ou un des deux) pattern n'est pas présent
                        ok = False
                # if it does, check its ID to match its target
                if ok:  # si le/les patterns sont présents, ajoute le sample au groupe de base/de reference
                    for i, id in enumerate(self.ids):
                        if id in s:
                            X.append(s)
                            y.append(self.targets[i])

            ##### Create the splits ######

            for i in range(self.nbr_splits):
                X_train, X_test, y_train, y_train = train_test_split(X, y, test_size=self.proportion, random_state=i)

                # replace pattern to match spliting
                # create target list accordingly

                pattern_list = first_patterns
                idx = len(self.pairings) - 1
                new_xy = []
                if idx + 1 == 1:  # if there is one pairing to do
                    new_xy = self._iterate_on_pattern_to_get_paired_files(pattern_list, idx, self.pairings[idx],
                                                                           X_train, X_test)
                elif idx + 1 == 2:  # if there is two pairings to do
                    X1 = X_train
                    X2 = X_test
                    for p in self.pairings[0]:
                        pattern_list[0] = p
                        new_sub_xy = self._iterate_on_pattern_to_get_paired_files(pattern_list, idx, self.pairings[idx],
                                                                                   X1, X2)
                        X1 = new_sub_xy[0][0]
                        X2 = new_sub_xy[0][1]

                        new_xy.extend(new_sub_xy)  # in shape of a list containing all [xtrain, xtest, ytrain, ytest] groups

                ###### save splits dans dict ######

                # shuffles
                new_xy = np.swapaxes(new_xy, 0, 1)  # is now in shape of 4 big lists (xtrain, xtest, ytrain, ytest)

                new_X_train = new_xy[0]
                new_y_train = new_xy[2]
                train_zip = list(zip(new_X_train, new_y_train))
                random.Random(13).shuffle(train_zip)
                X_train, y_train = zip(*train_zip)

                new_X_test = new_xy[1]
                new_y_test = new_xy[3]
                test_zip = list(zip(new_X_test, new_y_test))
                random.Random(13).shuffle(test_zip)
                X_test, y_test = zip(*test_zip)

                # save to dict
                X_train = [self.names_dict[i] for i in X_train]
                X_test = [self.names_dict[i] for i in X_test]

                self.dict_splits["split{}".format(i)] = [X_train, X_test, y_train, y_test]

    def _is_there_pairing_to_do(self):
        no_pairing = 0
        for p in self.pairings:
            if p == "no":
                no_pairing += 1
        if no_pairing == len(self.pairings):
            return False
        else:
            return True

    def _iterate_on_pattern_to_get_paired_files(self, base, idx, pairing, X1, X2):
        """

        :param base: list being updated of the patterns
        :param idx: index of the pairing to fit with base shape
        :param pairing: list of patterns for a single pairing
        :param X1: Xtrain data to replace pattern in
        :param X2: Xtest data to replace pattern in
        :return: lists of new xtrain, xtest, ytrain and ytest
        """
        p1 = base[idx]
        new_xy = []
        for pattern in pairing:
            base[idx] = pattern
            X1_2, X2_2, y1_2, y2_2 = self._replace_pattern_get_matching_files(p1, base[idx], X1, X2)
            new_xy.append([X1_2, X2_2, y1_2, y2_2])

        return new_xy

    def _replace_pattern_get_matching_files(self, p1, p2, Xtrain, Xtest):
        Xtrain_2 = [i.replace(p1, p2) for i in Xtrain]
        Xtest_2 = [i.replace(p1, p2) for i in Xtest]
        ytrain_2 = []
        ytest_2 = []

        for file in Xtrain_2:
            for i, id in enumerate(self.ids):
                if id in file.split("/")[-1]:
                    ytrain_2.append(self.targets[i])
        for file in Xtest_2:
            for i, id in enumerate(self.ids):
                if id in file.split("/")[-1]:
                    ytest_2.append(self.targets[i])

        return Xtrain_2, Xtest_2, ytrain_2, ytest_2

    def _test_pairing_patterns_spelling(self):
        """
        If the list obtained directly from the directory with a glob.glob is different of the list obtained by replacing
        the pattern from the "base list" (list of file containing the first pattern of the pairing) it means there is a
        problem of pattern spelling that will lead to further trouble in the rest of the processing
        :return: nothing
        """
        path_of_files = self.files.split("/")[:-1]
        path_of_files = "/".join(path_of_files)

        prob_patterns = []

        # TODO
        # not sure if all the cases are covered, especially the case where the pattern is not found and return the
        # same list unchanged
        for pairing in self.pairings:
            p1 = glob.glob(path_of_files + "*" + pairing[0] + "*")
            for pattern in pairing:
                p = glob.glob(path_of_files + "*" + pattern + "*")
                p_by_replace = [f.replace(pairing[0], pattern) for f in p1]

                if p.sort() != p_by_replace.sort():
                    prob_patterns.append(pattern)

        if len(prob_patterns) > 0:
            raise("There is a pattern error, here are the problematic patterns : {}".format(prob_patterns))



# #Conditionnal statement to handle split creation with pairing(s)
# if pairing_pn == "no" and pairing_12 == "no":  # no pairing
#     X = []
#     y = []
#     for file in files_list:
#         for i, id in enumerate(uniq_ID):
#             if id in file.split("/")[-1]:
#                 X.append(file)
#                 y.append(targets[i])
#
#     # Create the splits
#     for i in range(nbr_splits):
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=percent_in_test, random_state=i)
#         splits_dict["split{}".format(i)] = [X_train, X_test, y_train, y_test]
#
# elif pairing_pn != "no" and pairing_12 == "no":  # pairing of positive and negative files
#     X_pos = []
#     y_pos = []
#     for file in files_list:
#         if pair_id_pos in file.split("/")[-1]:
#             for i, id in enumerate(uniq_ID):
#                 if id in file.split("/")[-1]:
#                     X_pos.append(file)
#                     y_pos.append(targets[i])
#
#     # Create the splits
#     for i in range(nbr_splits):
#         X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_pos, y_pos, test_size=percent_in_test,
#                                                                     random_state=i)
#         X_train_n = [i.replace(pair_id_pos, pair_id_neg) for i in X_train_p]
#         X_test_n = [i.replace(pair_id_pos, pair_id_neg) for i in X_test_p]
#         y_train_n = []
#         y_test_n = []
#         for file in X_train_n:
#             for i, id in enumerate(uniq_ID):
#                 if id in file.split("/")[-1]:
#                     y_train_n.append(targets[i])
#         for file in X_test_n:
#             for i, id in enumerate(uniq_ID):
#                 if id in file.split("/")[-1]:
#                     y_test_n.append(targets[i])
#
#         # shuffle datasets to make sure pos files and neg are seen in a random order by the algorithms
#         X_train = X_train_p + X_train_n
#         y_train = y_train_p + y_train_n
#         train_zip = list(zip(X_train, y_train))
#         random.Random(13).shuffle(train_zip)
#         X_train, y_train = zip(*train_zip)
#
#         X_test = X_test_p + X_test_n
#         y_test = y_test_p + y_test_n
#         test_zip = list(zip(X_test, y_test))
#         random.Random(13).shuffle(test_zip)
#         X_test, y_test = zip(*test_zip)
#
#         splits_dict["split{}".format(i)] = [X_train, X_test, y_train, y_test]
#
# elif pairing_pn == "no" and pairing_12 != "no":  # pairing over another condition
#     X_1 = []
#     y_1 = []
#     for file in files_list:
#         if pair_id_1 in file.split("/")[-1]:
#             for i, id in enumerate(uniq_ID):
#                 if id in file.split("/")[-1]:
#                     X_1.append(file)
#                     y_1.append(targets[i])
#
#     # Create the splits
#     for i in range(nbr_splits):
#         X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size=percent_in_test,
#                                                                     random_state=i)
#         X_train_2 = [i.replace(pair_id_1, pair_id_2) for i in X_train_1]
#         X_test_2 = [i.replace(pair_id_1, pair_id_2) for i in X_test_1]
#         y_train_2 = []
#         y_test_2 = []
#         for file in X_train_2:
#             for i, id in enumerate(uniq_ID):
#                 if id in file.split("/")[-1]:
#                     y_train_2.append(targets[i])
#         for file in X_test_2:
#             for i, id in enumerate(uniq_ID):
#                 if id in file.split("/")[-1]:
#                     y_test_2.append(targets[i])
#
#         # shuffle datasets to make sure pos files and neg are seen in a random order by the algorithms
#         X_train = X_train_1 + X_train_2
#         y_train = y_train_1 + y_train_2
#         train_zip = list(zip(X_train, y_train))
#         random.Random(13).shuffle(train_zip)
#         X_train, y_train = zip(*train_zip)
#
#         X_test = X_test_1 + X_test_2
#         y_test = y_test_1 + y_test_2
#         test_zip = list(zip(X_test, y_test))
#         random.Random(13).shuffle(test_zip)
#         X_test, y_test = zip(*test_zip)
#
#         splits_dict["split{}".format(i)] = [X_train, X_test, y_train, y_test]
#
# elif pairing_pn != "no" and pairing_12 != "no":  # pairing pos/neg AND another condition
#     X_pos_1 = []
#     y_pos_1 = []
#     for file in files_list:
#         if pair_id_pos in file.split("/")[-1] and pair_id_1 in file.split("/")[-1]:
#             for i, id in enumerate(uniq_ID):
#                 if id in file.split("/")[-1]:
#                     X_pos_1.append(file)
#                     y_pos_1.append(targets[i])
#
#     # Create the splits
#     for i in range(nbr_splits):
#         X_train_pos_1, X_test_pos_1, y_train_pos_1, y_test_pos_1 = train_test_split(X_pos_1, y_pos_1,
#                                                                                     test_size=percent_in_test,
#                                                                                     random_state=i)
#         X_train_pos_2 = [i.replace(pair_id_1, pair_id_2) for i in X_train_pos_1]
#         X_test_pos_2 = [i.replace(pair_id_1, pair_id_2) for i in X_test_pos_1]
#         y_train_pos_2 = []
#         y_test_pos_2 = []
#         for file in X_train_pos_2:
#             for i, id in enumerate(uniq_ID):
#                 if id in file.split("/")[-1]:
#                     y_train_pos_2.append(targets[i])
#         for file in X_test_pos_2:
#             for i, id in enumerate(uniq_ID):
#                 if id in file.split("/")[-1]:
#                     y_test_pos_2.append(targets[i])
#
#         X_train_neg_1 = [i.replace(pair_id_pos, pair_id_neg) for i in X_train_pos_1]
#         X_test_neg_1 = [i.replace(pair_id_pos, pair_id_neg) for i in X_test_pos_1]
#         y_train_neg_1 = []
#         y_test_neg_1 = []
#         for file in X_train_neg_1:
#             for i, id in enumerate(uniq_ID):
#                 if id in file.split("/")[-1]:
#                     y_train_neg_1.append(targets[i])
#         for file in X_test_neg_1:
#             for i, id in enumerate(uniq_ID):
#                 if id in file.split("/")[-1]:
#                     y_test_neg_1.append(targets[i])
#
#         X_train_neg_2 = [i.replace(pair_id_1, pair_id_2) for i in X_train_neg_1]
#         X_test_neg_2 = [i.replace(pair_id_1, pair_id_2) for i in X_test_neg_1]
#         y_train_neg_2 = []
#         y_test_neg_2 = []
#         for file in X_train_neg_2:
#             for i, id in enumerate(uniq_ID):
#                 if id in file.split("/")[-1]:
#                     y_train_neg_2.append(targets[i])
#         for file in X_test_neg_2:
#             for i, id in enumerate(uniq_ID):
#                 if id in file.split("/")[-1]:
#                     y_test_neg_2.append(targets[i])
#
#         # shuffle datasets to make sure pos files and neg are seen in a random order by the algorithms
#         X_train = X_train_pos_1 + X_train_pos_2 + X_train_neg_1 + X_train_neg_2
#         y_train = y_train_pos_1 + y_train_pos_2 + y_train_neg_1 + y_train_neg_2
#         train_zip = list(zip(X_train, y_train))
#         random.Random(13).shuffle(train_zip)
#         X_train, y_train = zip(*train_zip)
#
#         X_test = X_test_pos_1 + X_test_pos_2 + X_test_neg_1 + X_test_neg_2
#         y_test = y_test_pos_1 + y_test_pos_2 + y_test_neg_1 + y_test_neg_2
#         test_zip = list(zip(X_test, y_test))
#         random.Random(13).shuffle(test_zip)
#         X_test, y_test = zip(*test_zip)
#
#         splits_dict["split{}".format(i)] = [X_train, X_test, y_train, y_test]