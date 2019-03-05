import os
import re
import pickle
import random
import numpy as np
import json
from collections import defaultdict

from tqdm import tqdm
from skimage.io import imread

from utils.dictionary import Dictionary


class Nsc_Data():
    def __init__(self, img_dir, split_path, 
    pickles_dir, vocab_min, subset_users, comm_min):
        self.img_dir = img_dir
        self.split_path = split_path
        self.pickles_dir = pickles_dir
        # Minimum number of times word must occur to be included in the vocab
        self.vocab_min = vocab_min
        self.subset_users = subset_users
        # Min number of comments for a post to be included
        self.comm_min = comm_min

    def __get_paths(self):
        ret_dict = {}
        paths_pickle = os.path.join(self.pickles_dir, "imgpaths.pickle")
        # Dumped into pickle for speed
        if os.path.exists(paths_pickle):
            with open(paths_pickle, "rb") as rf:
                ret_dict = pickle.load(rf)
        else:
            for root, _, files in os.walk(self.img_dir):
                if len(files) != 0:
                    for fl in files:
                        path = os.path.join(root, fl)
                        img_id = "".join(re.findall(r"[0-9]+", root))
                        ret_dict[img_id] = path
            with open(paths_pickle, "wb") as wf:
                pickle.dump(ret_dict, wf)
        return ret_dict

    def __split_img_data(self, paths, train_split_f, val_split_f, test_split_f):
        train, val, test = [], [], []
        for split,split_f in (
            (train, train_split_f), (val, val_split_f), (test, test_split_f)):
            with open(split_f) as rf:
                for line in rf:
                    line = line.split(".")[0]
                    split.append(paths[line])
        return train, val, test

    def __prepare_comments(self, comments):
        comments = json.loads(comments)
        comm_d, comms = {}, []
        for com in comments:
            for attr in com:
                attr_name, attr_value = tuple(attr.items())[0]
                # Preprocess
                if attr_name == "Url":
                    attr_value = re.sub(r"https?://", "", attr_value)
                    attr_value = re.sub(r"\"", "", attr_value)
                    attr_value = [
                        word.lower() for word in attr_value if word not in 
                        "“”….''"]
                    attr_value = "".join(attr_value)
                elif attr_name == "Comment":
                    attr_value = re.sub(
                        r'([\“\…\”\'\"\.\(\)\!\?\-\\\/\,\:❤️♥❤])',
                        r' \1 ',
                        attr_value)
                    attr_value = re.sub("<3*", "<3", attr_value)
                    attr_value = re.sub("[❤️♥❤]", "<love>", attr_value)
                    attr_value = attr_value.replace('&', ' and ')
                    attr_value = attr_value.replace('@', ' at ')
                    attr_value = attr_value.strip().split(" ")
                    # Filter out non-acii
                    comm_prepr = []
                    for wd in attr_value:
                        try:
                            wd = wd.encode('ascii').strip()
                            wd = wd.decode("ascii")
                            comm_prepr.append(wd)
                        except:
                            pass
                    attr_value = comm_prepr
                    if len(attr_value) == 0:
                        break  # Skip non-valid non-english comment
                    attr_value = [
                        word.lower() for word in attr_value if word not in 
                        "“”….''"]
                    attr_value = " ".join(attr_value)
                comm_d[attr_name] = attr_value
            if {"Name", "Url", "Comment"} == set(comm_d.keys()):
                comms.append(comm_d)
                comm_d = {}
        return comms
    
    def __prepare_txt_data(self):
        # Commnents: [{"Comment":...}, ...], HashTag, PostName, PosterID
        p_attr_list = ["PosterID", "Comments"]
        u_attr_list = []
        # Read the other needed attributes: pickles_dir/ttributes.txt
        # attributes.txt in form [USR]atr1,...[POSTS]attr1,..
        with open(os.path.join(self.pickles_dir, "attributes.txt")) as rf:
            attrlist = None
            for line in rf:
                line = line.strip()
                if line == "[USR]":
                    attrlist = u_attr_list
                    continue
                elif line == "[POST]":
                    attrlist = p_attr_list
                    continue
                if "#" in line:  # Comments
                    continue
                attrlist.append(line)
        print(p_attr_list)
        print(u_attr_list)
        posts_path = os.path.join(self.pickles_dir, "posts_nsc.pickle")
        usrs_path = os.path.join(self.pickles_dir, "users_nsc.pickle")
        with open(posts_path, "rb") as rf:
            posts_data = pickle.load(rf)
        with open(usrs_path, "rb") as rf:
            usrs_data = pickle.load(rf)
        attr_dict = {}  # {"PostID": {attr1:..., attr2:...}}
        skiped = 0
        for post_id in tqdm(list(posts_data.keys())):
            pattr_d = {}  # Post attributes
            usrattr_d = {}  # User attributes
            post_data = posts_data[post_id]
            skip = False
            for pattr in p_attr_list:
                if pattr == "Comments":
                    # Some data no comments, skip
                    comments = post_data[pattr]
                    if not comments:
                        skip = True
                        skiped += 1
                        break
                    comments = self.__prepare_comments(comments)
                    # Dont add if less than comm_min comments
                    if len(comments) < self.comm_min:
                        skiped += 1
                        skip = True
                        break  # Skip it
                    pattr_d[pattr] = comments
                elif pattr == "PosterID":
                    # Get Poster attributes
                    poster = post_data[pattr]
                    pattr_d[pattr] = poster
                    user_data = usrs_data[poster]
                    for usrattr in u_attr_list:
                        # Process Info
                        if usrattr == "Info":
                            info = user_data[usrattr]
                            info = re.sub(r"歲", "y.o.", info)
                            info = re.sub(r"來自", "from", info)
                            info = re.sub(r"女性", "female", info)
                            info = re.sub(r"男性", "male", info)
                            usrattr_d[usrattr] = info
                        elif usrattr == "Country":
                            country = user_data[usrattr]
                            country = re.sub(r" ", "_", country)
                            usrattr_d[usrattr] = country
                        else:
                            usrattr_d[usrattr] = user_data[usrattr]
                elif pattr == "PostName":
                    postname = post_data[pattr]
                    pattr_d[pattr] = [postname]
                else:
                    pattr_d[pattr] = json.loads(post_data[pattr])
            if skip == False:
                attr_dict[post_id] = {
                    "PostAttr": pattr_d,
                    "UsrAttr": usrattr_d}
        print(
            "Skipped no or less than {} comments: ".format(self.comm_min), 
            skiped)
        self.p_attr_list = attr_dict
        self.u_attr_list = u_attr_list
        return attr_dict

    def __gen_subset(self, attr_data, train_spl, val_spl, test_spl):
        com_count = defaultdict(int)
        for postid in tqdm(attr_data):
            comments = attr_data[postid]["PostAttr"]["Comments"]
            for comm in comments:
                com_count[comm["Url"]]+= 1
        com_count = sorted(com_count.items(), key=lambda x: -x[1])
        # Choose subset
        subset,_ = zip(*com_count)
        subset = set(subset[:self.subset_users])
        retdict = {"train": [], "val": [], "test": []}
        for spname,split in (
            ("train", train_spl), ("val", val_spl), ("test", test_spl)):
            for imgpath in split:
                postid = self.paths_postid[imgpath]
                try:
                    comments = attr_data[postid]["PostAttr"]["Comments"]
                except:
                    # Skiped image
                    continue
                comm_list = []
                for comm in comments:
                    if comm["Url"] in subset:
                        comm_list.append((comm["Url"], comm["Comment"]))
                if len(comm_list) < self.comm_min:
                    continue
                else:
                    # # Decided to add, dont forget to add more attributes
                    all_attrib = attr_data[postid]
                    other_attrib = {
                        "UsrAttr": all_attrib["UsrAttr"],
                        "PostAttr": {}}
                    for attr in all_attrib["PostAttr"]:
                        # We dont need some attrs
                        if attr not in ("Comments", "PosterID"):
                            other_attrib["PostAttr"][
                                attr] = all_attrib["PostAttr"][attr]
                    comm_list = {imgpath: comm_list}
                    # Add other atributes in case of futher need
                    retdict[spname].append((comm_list, other_attrib))
        print("Subset statistics:")
        print("Train data: {} \nVal data: {} \nTest data: {}".format(
            len(retdict["train"]), len(retdict["val"]), len(retdict["test"])))
        return retdict["train"], retdict["val"], retdict["test"]

    def __tokenize(self, train, val, test, comm_maxlen=None, pname_maxlen=None):
        # train, val, test - image splits read fromfile
        # data in form [({'img_path': "Comments"}, OtherAttribsDict)]
        # Attrdict example: 
        #{'PostAttr': [{'HashTag': {}}], 'UsrAttr': 
        # {'Age': None, 'Country': 'Australia', 'UserName': 'Chloe T'}}
        self.dictionaries = {}
        # Prepare comments)
        urls, comms = [], []
        other_attrs = defaultdict(list)
        for imidc, otherattr in train:
            urls_, comms_ = zip(*list(imidc.items())[0][1])
            urls.extend(urls_)
            comms.extend(comms_)
            p_attr, u_attr = otherattr["PostAttr"], otherattr["UsrAttr"]
            for attr in p_attr:
                pattr = p_attr[attr]
                other_attrs[attr].extend(pattr)
                other_attrs[attr] = list(set(other_attrs[attr]))
            for uattr in u_attr:
                # User attrs
                other_attrs[uattr].append(u_attr[uattr])
                other_attrs[uattr] = list(set(other_attrs[uattr]))
        self.dictionaries["Comments"] = Dictionary(self.vocab_min)
        self.dictionaries["Comments"].fit(comms)
        self.dictionaries["Urls"] = Dictionary()
        self.dictionaries["Urls"].fit(urls)
        # Other attributes
        for attr in other_attrs:
            self.dictionaries[attr] = Dictionary()
            self.dictionaries[attr].fit(other_attrs[attr])
        # Transform in advance for the better performance
        train_tr, val_tr, test_tr = [], [], []
        sets = [train_tr, val_tr, test_tr]
        for i,dataset in enumerate([train, val, test]):
            for imidc, otherattr in dataset:
                imid = list(imidc.items())[0][0]
                urls_, comms_ = zip(*list(imidc.items())[0][1])
                urls_ = self.dictionaries["Urls"].transform(urls_)
                comms_ = self.dictionaries["Comments"].transform(
                    comms_, True, max_len=comm_maxlen)
                oattr_t = {}  # Transform and combine attributes
                p_attr, u_attr = otherattr["PostAttr"], otherattr["UsrAttr"]
                for pattr in p_attr:
                    pattr_ = p_attr[pattr]
                    if len(pattr_) == 0:
                        pattr_ = self.dictionaries[pattr].unk_token
                    # NOTE: think how to make it more universal for sequence attrs
                    if pattr == "PostName":
                        pattr_ = self.dictionaries[pattr].transform(
                            pattr_, True, max_len=pname_maxlen)
                    else:
                        pattr_ = self.dictionaries[pattr].transform(pattr_)
                    oattr_t[pattr] = pattr_
                for uattr in u_attr:
                    uattr_ = u_attr[uattr]
                    if len(uattr_) == 0:
                        uattr_ = [self.dictionaries[uattr].unk_token]
                    uattr_ = self.dictionaries[uattr].transform(uattr_)
                    oattr_t[uattr] = uattr_
                trans_data = (imid, urls_, comms_, oattr_t)
                sets[i].append(trans_data)
        return train_tr, val_tr, test_tr

    def __prepare_data(self):
        # Paths dict in form {PostID: image_path}
        print("Getting paths")
        img_paths = self.__get_paths()
        # Get reversed dict, later use for batch generation
        self.paths_postid = {path: postid for (
            postid, path) in img_paths.items()}
        # Get splits from NSC paper split
        train_split_f = os.path.join(self.split_path, "train.txt")
        val_split_f = os.path.join(self.split_path, "val.txt")
        test_split_f = os.path.join(self.split_path, "test.txt")
        # Get Images data split in form [path1, path2, ...]
        train_spl, val_spl, test_spl = self.__split_img_data(
            img_paths, train_split_f, val_split_f, test_split_f)
        print("All data: ", len(img_paths.keys()))
        print("Train data: {} \nVal data: {} \nTest data: {}".format(
            len(train_spl), len(val_spl), len(test_spl)))
        # Load posts and users data from prepared .pickle files
        # attr_data = dict({post_id: [{PostAttr: {...}}, {UsrAttr: {...}}]})
        print("Prepare attributes")
        attr_data = self.__prepare_txt_data()
        print(
            "Create subset of {} popular commentators comments".format(
                self.subset_users))
        # Final variant of data
        train, val, test = self.__gen_subset(
            attr_data, train_spl, val_spl, test_spl)
        return train, val, test
    
    def prepare_data(
        self, 
        load_prepared=False, prepared_path=None, 
        comm_maxlen=None, pname_maxlen=None,
        annot_folder=None, annot_prepare=False,
        annot_numref=4):
        if not prepared_path:
            prepared_path = os.path.join(
                self.pickles_dir, "data_prepared.pickle")
        if load_prepared and os.path.exists(prepared_path):
            with open(prepared_path, "rb") as rf:
                train, val, test = pickle.load(rf)
            print("Subset statistics:")
            print("Train data: {} \nVal data: {} \nTest data: {}".format(
            len(train), len(val), len(test)))
        else:
            train, val, test = self.__prepare_data()
            with open(prepared_path, "wb") as wf:
                pickle.dump((train, val, test), wf)
        # Save annot.json
        if (not load_prepared) or annot_prepare:
            if not annot_folder:
                raise ValueError("Must define annotaion folder")
            self.__prepare_annot(train, val, test, annot_numref, annot_folder)
        # Tokenize, get dicts
        self.train, self.val, self.test = self.__tokenize(
            train, val, test, comm_maxlen, pname_maxlen)
    
    def save_dicts(self, dicts_pickle):
        with open(dicts_pickle, "wb") as wf:
            pickle.dump(self.dictionaries, wf)
    
    def __prepare_annot(self, train, val, test, num_ref, annot_folder):
        # num_ref: number of reference captions
        if not os.path.exists(annot_folder):
            os.makedirs(annot_folder)
        print("Preparing ground truth annotations")
        sets = ("train", "val", "test")
        def dump_to_json(obj, f_name):
            path = os.path.join(annot_folder, f_name)
            if os.path.exists(path):
                print("")  # Need some pause or problem with deletion
                os.remove(path)
            with open(path, 'w') as wf:
                json.dump(obj, wf)
            print("Saved into :", path)

        for ii, dataset in enumerate((train, val, test)):
            eval_d_list = []
            img_info_list = []
            print(len(dataset))
            for imidc, _ in dataset:
                imid_path = list(imidc.items())[0][0]
                imid = "".join(re.findall("[0-9]", imid_path))
                _, comms_ = zip(*list(imidc.items())[0][1])
                # Randomly choose 4 captions
                indices = np.random.permutation(len(comms_))[:num_ref]
                # Need some preprocessing
                comms_prepr = []
                for comm in comms_:
                    comm = comm.split(" ")
                    comm_prepr = []
                    for com in comm:
                        try:
                            com = com.encode('ascii').strip()
                            com = com.decode("ascii")
                            comm_prepr.append(com)
                        except:
                            pass
                    comm = comm_prepr
                    comms_prepr.append(" ".join(comm))
                comms_ = comms_prepr
                comms_ = list(np.array(comms_)[indices])
                for i in range(len(comms_)):
                    ev_dict = {
                        'image_id': int(imid),
                        'caption': comms_[i],
                        'id': int(imid)}
                    eval_d_list.append(ev_dict)
                im_dict = {
                    'id' : int(imid),
                    'file_name': imid_path}
                img_info_list.append(im_dict)
            annot_dict = {
                'annotations': eval_d_list,
                'images': img_info_list,
                'type': 'captions',
                'licenses': [{}],
                'info': {'description': "NSC Lookbook Evaluation"}}
            dump_to_json(
                annot_dict,
                sets[ii] + ".json")
        

class Data():
    """General data class, generating batch, padding"""
    def __init__(
        self, data, dictionaries, n_comments,
        max_len_comm=None, max_len_post=None):
        # Data in form [(imgpath, urls, comments, )]
        self._data = data
        # dictioray attribute of data class (e.g. Nsc_data)
        self._dicts = dictionaries
        self._n_comments = n_comments  # Number of comms in batch
        # Temporary solution, for using static RNNs
        self._max_len_comm = max_len_comm
        self._max_len_post = max_len_post
    
    def batch_generator(self, batch_size, shuffle=True):
        data_len = len(self._data)
        if shuffle:
            order = np.random.permutation(data_len)
        else:
            order = np.arange(data_len)

        n_batches = data_len // batch_size
        # if data_len % batch_size:
        #     n_batches += 1

        for k in range(n_batches):
            batch_start = k * batch_size
            batch_end = min((k + 1) * batch_size, data_len)
            # current_batch_size = batch_end - batch_start
            data_cur = []
            max_len_comm = 0
            max_len_post = 0
            for idx in order[batch_start: batch_end]:
                impath, urls, comms, other_attrs = self._data[idx]
                impath_tiled = np.tile(np.array(impath), 1)
                # Select urls, comms pairs
                indices = np.random.permutation(len(urls))[:self._n_comments]
                urls = np.array(urls)[indices]
                comms = np.array(comms)[indices]
                # find max commment length
                if self._max_len_comm is not None:
                    max_len_comm = self._max_len_comm
                else:
                    max_len_comm = max(max_len_comm, len(max(comms, key=len)))
                # TODO: add proccessing for sequence attributess
                pname = other_attrs["PostName"]
                if self._max_len_post is not None:
                    max_len_post = self._max_len_post
                else:
                    max_len_post = max(max_len_post, len(max(pname, key=len)))
                data_cur.append((impath_tiled, urls, comms, pname, other_attrs))
            data_ = []
            for impath, urls, comms, pname, other_attr in data_cur:
                # Zero padding for comments
                x = np.ones(
                    [self._n_comments, max_len_comm], 
                    dtype=np.int32) * self._dicts["Comments"].word2idx["<PAD>"]
                y = x.copy()
                lengths = np.zeros(self._n_comments, dtype=np.int32)
                for n in range(self._n_comments):
                    utt_len = len(comms[n]) - 1
                    x[n, :utt_len] = comms[n][:-1]
                    lengths[n] = utt_len
                    y[n, :utt_len] = comms[n][1:]
                # Zero padding for pnames
                px = np.ones(
                    [1, max_len_post], 
                    dtype=np.int32) * self._dicts["PostName"].word2idx["<PAD>"]
                pxlen = len(pname[0]) - 1
                px[0, :pxlen] = pname[0][:-1]
                px = np.tile(px, (self._n_comments, 1))
                data_.append((impath, urls, x, y, lengths, px, other_attr))
            # Stack impaths x, y, lengths
            impaths, urls, xs, ys, lengths, pn, other_attrs = zip(*data_)
            ret_tuple = (
                np.concatenate(impaths),
                np.squeeze(np.concatenate(urls)),
                (np.concatenate(xs), 
                np.concatenate(ys), 
                np.concatenate(lengths)),
                np.concatenate(pn),
                other_attrs)
            yield ret_tuple
    
    @property
    def dictionaries(self):
        return self._dicts