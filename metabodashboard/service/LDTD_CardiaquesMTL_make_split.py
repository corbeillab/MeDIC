from __future__ import print_function, division, absolute_import, unicode_literals

# from sklearn.externals.six.moves import range
from msvlm.msvlm.identification import VirtualLockMassCorrector

# import msvlm.msAlign.msAlign as ms
import preprocessing.RFMSA2_old_new
import os, sys, glob

# import h5py as h
from .metabodashboard.service.pymspec.spectrum import unify_mz
from .metabodashboard.service.pymspec.io.ion_list.file_loader import *
from preprocessing.Utils import *
import copy

# from vlm.identification import VirtualLockMassCorrector
import pickle as pkl
import datetime


def select_n_sample_random(spectra, number):
    g1 = copy.copy(spectra)
    g2 = []
    while len(g2) != number:
        index = np.random.randint(0, len(g1) - 1)
        g2.append(g1.pop(index))
    return g1, g2


def sep_train_test(spectra, percent):  # list of spect

    base = []
    extract = []
    for i, j in enumerate(spectra):
        if "CTL" in j:
            extract.append(j)
        else:
            base.append(j)

    g1 = base
    g2 = []
    while len(g2) < len(g1) * percent:
        index = np.random.randint(0, len(g1) - 1)
        g2.append(g1.pop(index))

    g2.extend(extract)
    return g1, g2


def create_one_split(input):
    # -------- Take inputs -------
    split_id = input[0]
    print("split handled : {}".format(split_id))
    targets = input[1]  # dict format where key : file name and value : target number
    files_list = list(input[2])  # Suppose files_list is randomised
    output_path = input[3]

    # ------ Constant used for pre-processing ------
    peak_threshold = 500
    # number_in_test = 72
    percent_in_test = 0.20
    number_for_autoOptimize = 20

    # ----- seperate pos from neg ----
    lcs_list_pos = []
    lcs_list_neg = []

    for (
        i
    ) in (
        files_list
    ):  # there was an error in labeling, so the pos and neg in file's name are not valid, we must check ionization with plate number
        if "plate01" in i or "plate02" in i:
            lcs_list_pos.append(i)
        elif "plate03" in i or "plate04" in i:
            lcs_list_neg.append(i)

            ################
    ##     Pos    ##
    ################
    print("--> Doing pos files")
    # ---------- Format Train vs Test ----------
    train_list, test_list = sep_train_test(lcs_list_pos, percent_in_test)

    spect_train = list(load_ion_list(train_list, mz_precision=4))
    spect_test = list(load_ion_list(test_list, mz_precision=4))
    print("------ spect_train len : {}".format(len(spect_train)))

    # ----- Create target metadata for each file -----
    for s in spect_train + spect_test:
        # print("s.metadata['file'] : {}".format(s.metadata["file"]))
        name = s.metadata["file"].split("/")[-1]
        s.metadata["target"] = targets[name]

    # ------ Check if file target exist ----
    for s in spect_train + spect_test:
        try:
            s.metadata["target"]
        except KeyError:
            print("Problem with targets. %s has no target" % s.metadata["file"])
            sys.exit(1)

    # ------ Preprocess data -----
    preprocessing_pipeline = common.Pipeline(
        [discrete.ThresholdedPeakFiltering(peak_threshold)]
    )
    test = list(preprocessing_pipeline.fit_transform(spect_test))
    train = list(preprocessing_pipeline.fit_transform(spect_train))

    print("Train pos: %s" % len(train))
    print("Test pos: %s" % len(test))

    print("Auto-optim num: %s" % number_for_autoOptimize)
    train_optimizer, rest = select_n_sample_random(train, number_for_autoOptimize)
    logging.debug("%s : Removing low intensity peaks" % split_id)

    # -------------- Do the VLM thing -----------

    vlm = VirtualLockMassCorrector(window_size=10, minimum_peak_intensity=1000)

    logging.debug("%s : VLM optimization" % split_id)
    vlm.optimize_window_size(train_optimizer)

    logging.debug("%s : VLM training" % split_id)
    vlm.fit(train)

    logging.debug("%s : VLM apply" % split_id)
    train = vlm.transform(train)
    test = vlm.transform(test)

    train_optimizer = vlm.transform(train_optimizer)

    # ----------- Do the RFMSA thing ------------
    rfmsa = preprocessing.RFMSA2_old_new.Reference_free_aligner2(min_mz=50, max_mz=1200)
    logging.debug("%s : RFMSA optimization" % split_id)
    rfmsa.autoOptimize(train_optimizer, max_distance_values=(40, 50, 60, 70, 80))

    logging.debug("%s : RFMSA train" % split_id)
    rfmsa.train(train)

    logging.debug("%s : RFMSA apply" % split_id)
    rfmsa.apply(train)
    rfmsa.apply(test)

    # Perform intensity normalisation
    # logging.debug("%s Intensity corretion" %split_id)
    # intensity_normalisation.apply(train)
    # intensity_normalisation.apply(test)

    # ------ Unify mz -----------
    logging.debug("%s : Unifying mz" % split_id)
    spect = list(train) + list(test)
    unify_mz(spect)

    ####################
    #        Neg      ##
    ####################

    print("--> Doing the neg files")
    # ---------- Format Train vs Test ----------
    train_list_neg = []
    for i in train_list:
        if "plate01" in i:
            train_list_neg.append(i.replace("plate01", "plate03"))
        elif "plate02" in i:
            train_list_neg.append(i.replace("plate02", "plate04"))

    test_list_neg = []
    for i in test_list:
        if "plate01" in i:
            test_list_neg.append(i.replace("plate01", "plate03"))
        elif "plate02" in i:
            test_list_neg.append(i.replace("plate02", "plate04"))

    # train_list_neg = [i.replace("pos", "neg") for i in train_list]
    # test_list_neg = [i.replace("pos", "neg") for i in test_list]

    if len(lcs_list_neg) == len(train_list_neg + test_list_neg):
        pass
    else:
        print("len de lcs_list_neg non correspondant avec train/test_list_neg")

    for i in lcs_list_neg:
        if i in train_list_neg + test_list_neg:
            pass
        else:
            print("lcs_list_neg non correspondant avec train/test_list_neg")

    spect_train_neg = list(load_ion_list(train_list_neg, mz_precision=4))
    spect_test_neg = list(load_ion_list(test_list_neg, mz_precision=4))

    # No need to create the metadata target since we merge neg with pos at the end, so they end up sharing
    # the previously created metadata for pos

    # ------------ Preprocess data -------------
    preprocessing_pipeline = common.Pipeline(
        [discrete.ThresholdedPeakFiltering(peak_threshold)]
    )
    test_neg = list(preprocessing_pipeline.fit_transform(spect_test_neg))
    train_neg = list(preprocessing_pipeline.fit_transform(spect_train_neg))

    print("Train neg: %s" % len(train_neg))
    print("Test neg: %s" % len(test_neg))

    print("Auto-optim num: %s" % number_for_autoOptimize)
    train_optimizer, rest = select_n_sample_random(train_neg, number_for_autoOptimize)
    logging.debug("%s : Removing low intensity peaks" % split_id)

    # ------------ Do the VLM thing -----------
    vlm_neg = VirtualLockMassCorrector(window_size=10, minimum_peak_intensity=1000)

    logging.debug("%s : VLM optimization" % split_id)
    vlm_neg.optimize_window_size(train_optimizer)

    logging.debug("%s : VLM training" % split_id)
    vlm_neg.fit(train_neg)

    logging.debug("%s : VLM apply" % split_id)
    train_neg = vlm_neg.transform(train_neg)
    test_neg = vlm_neg.transform(test_neg)

    train_optimizer = vlm_neg.transform(train_optimizer)

    # ----------- Do the RFMSA thing ------------
    rfmsa_neg = preprocessing.RFMSA2_old_new.Reference_free_aligner2(
        min_mz=50, max_mz=1200
    )
    logging.debug("%s : RFMSA optimization" % split_id)
    rfmsa_neg.autoOptimize(train_optimizer, max_distance_values=(40, 50, 60, 70, 80))

    logging.debug("%s : RFMSA train" % split_id)
    rfmsa_neg.train(train_neg)

    logging.debug("%s : RFMSA apply" % split_id)
    rfmsa_neg.apply(train_neg)
    rfmsa_neg.apply(test_neg)

    # Perform intensity normalisation
    # logging.debug("%s Intensity corretion" %split_id)
    # intensity_normalisation.apply(train)
    # intensity_normalisation.apply(test)

    # --------- Unify mz -----------
    logging.debug("%s : Unifying mz" % split_id)
    spect_neg = list(train_neg) + list(test_neg)
    unify_mz(spect_neg)

    # ---  Multiply m/z values by -1 in order to have distinct features ----
    for s in spect_neg:
        s.set_peaks(s.mz_values * -1, s.intensity_values)

    ###########
    #  Merge! #
    ###########

    merged_spect = []
    for i, s in enumerate(spect):
        n = spect_neg[i]
        merged_spect.append(
            Spectrum(
                np.append(s.mz_values, n.mz_values),
                np.append(s.intensity_values, n.intensity_values),
                mz_precision=4,
                metadata=s.metadata,
            )
        )

    ####################
    #     Output       #
    ####################
    print("--> Creating the output")
    dataset = [i.intensity_values for i in merged_spect]
    targets = [i.metadata["target"] for i in merged_spect]
    # samples_name = [str(i.metadata["file"]) for i in merged_spect]

    # vlm_parameters = [vlm.window_size, vlm.minimum_peak_intensity]

    # rfmsa_parameters = [rfmsa.min_mz, rfmsa.max_mz, rfmsa.max_distance]

    logging.debug("%s : Writing data to disk" % split_id)
    n_training_examples = len(train)
    design_name = "Ctrl_vs_Case"

    outpath = os.path.join("Splits", "{}_{}".format(design_name, split_id))
    print("outpath : {}".format(outpath))

    with open(outpath, "wb") as fo:
        pkl.dump(dataset[:n_training_examples], fo)  # train dataset
        pkl.dump(targets[:n_training_examples], fo)  # train targets
        pkl.dump(dataset[n_training_examples:], fo)  # test dataset
        pkl.dump(targets[n_training_examples:], fo)  # test targets

    # f = h.File("/home/fraeli01/Projet_LDTD_FluA/splits/Split_"+str(split_id)+"_LDTD_FluA_newAnalysis.h5", 'w')
    # logging.debug("%s : Writing data to disk" % split_id)
    # f.create_dataset("data", data=dataset)

    # logging.debug("%s : Writing targets to disk" % split_id)
    # f.create_dataset("target", data=targets)

    # logging.debug("%s : Writing fit_parameters to disk" % split_id)
    # f.create_dataset("n_training_examples", data=n_training_examples)

    # logging.debug("%s : Writing VLM parameters to disk" % split_id)
    # f.create_dataset("vlm_parameters", data=vlm_parameters)
    # f.create_dataset("vlm_points", data=vlm._vlm_mz)

    # logging.debug("%s : Writing RFMSA parameters to disk" % split_id)
    # f.create_dataset("rfmsa_parameters", data=rfmsa_parameters)
    # f.create_dataset("rfmsa_points", data=rfmsa.reference_mz)

    # f.create_dataset("features_list", data=merged_spect[0].mz_values)
    # f.create_dataset("samples_name", data=samples_name) # Needed to evaluate the performances on each split individually
    # f.close()

    print("split " + str(split_id) + " fini !")


def shuffle_file_names(files):
    index = np.arange(len(files))
    np.random.shuffle(index)
    return files[index]


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG, format="%(asctime)s.%(msecs)d %(levelname)s %(module)s - %(process)d - %(
    # funcName)s: %(message)s")

    path = ""  # /is2/projects/JC_Elinaf/Splits_cardiaque_Mtl/ (sur ls33)
    print(path)
    lcs_list = np.array(glob.glob(path + "LCS/*.lcs"))
    targets = {}
    for file_name in lcs_list:
        file_name = file_name.split("/")[-1]
        file_name = file_name.split(".")[0]
        date, archi, plate, pos_neg, code, no_id = file_name.split("_")
        t = 0
        if "C" in code:
            t = 1
        targets[file_name + ".lcs"] = t

    print(len(targets))

    number_of_split = 40
    jobs = []

    # Randomized and create job
    for i in range(number_of_split):
        randomized_list = shuffle_file_names(lcs_list)
        jobs.append((i, targets, randomized_list, path))
    # print("%s, %s" %(i, randomized_list))
    logging.debug("Created %s jobs. Now preparing pickle files." % len(jobs))

    start = datetime.datetime.now()
    print(start)
    create_one_split(jobs[0])
    print("One job took : {}".format(datetime.datetime.now() - start))

#    pool = Pool(processes=2) # Could be set otherwise.
#    pool.map(create_one_split, jobs)
#    pool.close()
#    pool.join()
