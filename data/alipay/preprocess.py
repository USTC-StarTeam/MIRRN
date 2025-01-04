import pandas as pd
import random
from tqdm import tqdm


def to_df(file_name):
    # df = pd.read_csv(file_name, header=None, names=['uid', 'iid', 'cid', 'btag', 'time'])
    # use_ID, sel_ID, ite_ID, cat_ID, act_ID, time
    df = pd.read_csv(file_name, header=0, names=['uid', 'sel_id', 'iid', 'cid', 'btag', 'time'])
    df = df[df['btag'] == 0]
    return df


def remap(df, max_len):
    # 特征id化 顺序：item user cate btag
    item_key = sorted(df['iid'].unique().tolist())
    item_len = len(item_key)
    item_map = dict(zip(item_key, range(1, item_len + 1)))
    df['iid'] = df['iid'].map(lambda x: item_map[x])
    # df['remapped_iid'], id_labels = pd.factorize(df['iid'])
    # df['remapped_iid'] = df['remapped_iid'] + 1
    # id_counts = df['remapped_iid'].value_counts().to_dict()
    # np.save('../data/taobao/popularity.npy', id_counts)

    user_key = sorted(df['uid'].unique().tolist())
    user_len = len(user_key)
    user_map = dict(zip(user_key, range(1, user_len + 1)))
    df['uid'] = df['uid'].map(lambda x: user_map[x])

    cate_key = sorted(df['cid'].unique().tolist())
    cate_len = len(cate_key)
    cate_map = dict(zip(cate_key, range(1, cate_len + 1)))
    df['cid'] = df['cid'].map(lambda x: cate_map[x])

    # btag_key = sorted(df['btag'].unique().tolist())
    # btag_len = len(btag_key)
    # btag_map = dict(zip(btag_key, range(1, btag_len + 1)))
    # df['btag'] = df['btag'].map(lambda x: btag_map[x])

    sel_key = sorted(df['sel_id'].unique().tolist())
    sel_len = len(sel_key)
    sel_map = dict(zip(sel_key, range(1, sel_len + 1)))
    df['sel_id'] = df['sel_id'].map(lambda x: sel_map[x])

    print("remap completed")
    print(f"The number of users:{user_len}")
    print(f"The number of items:{item_len}")
    print(f"The number of cates:{cate_len}")
    print(f"The number of sellers:{sel_len}")
    # print(f"The number of btags:{btag_len}")
    # 626041, 2200291, 72, 9999
    return df, user_len, item_len, cate_len


def gen_user_item_group(df):
    # 根据uid、time排序， uid分组
    # 根据iid、time排序， iid分组
    user_df = df.sort_values(['uid', 'time']).groupby('uid')
    item_df = df.sort_values(['iid', 'time']).groupby('iid')
    # cate_df = df.sort_values((['cid'])).groupby('cid')
    print("group completed")
    return user_df, item_df


def gen_dataset(user_df, item_df, max_len, min_len):
    train_label_array = []
    train_uid_array = []
    train_iid_array = []
    train_cate_array = []
    train_hist_iid_array = []
    train_hist_cate_array = []

    valid_label_array = []
    valid_uid_array = []
    valid_iid_array = []
    valid_cate_array = []
    valid_hist_iid_array = []
    valid_hist_cate_array = []

    test_label_array = []
    test_uid_array = []
    test_iid_array = []
    test_cate_array = []
    test_hist_iid_array = []
    test_hist_cate_array = []

    # get each user's last touch point time
    user_last_touch_time = []
    for uid, hist in user_df:
        user_last_touch_time.append(hist['time'].tolist()[-1])

    # uid - 1: last_time
    sorted_with_indices = sorted(enumerate(user_last_touch_time), key=lambda x: x[1])
    # uid : last_time
    sorted_indices = [x[0] + 1 for x in sorted_with_indices]
    split1, split2 = int(len(sorted_indices) * 0.8), int(len(sorted_indices) * 0.9)
    train_valid_test_flag = {uid: 1 for uid in sorted_indices[: split1]}
    train_valid_test_flag.update({uid: 2 for uid in sorted_indices[split1: split2]})
    train_valid_test_flag.update({uid: 3 for uid in sorted_indices[split2:]})

    cnt = 0
    for uid, hist in tqdm(user_df):
        if len(hist) < min_len:
            continue
        cnt += 1
        item_hist = hist['iid'].tolist()
        cate_hist = hist['cid'].tolist()
        # the item history part of the sample
        if len(item_hist) > max_len:
            cate_hist = cate_hist[len(item_hist) - max_len:]
            item_hist = item_hist[len(item_hist) - max_len:]
        pos_item = item_hist[-1]
        pos_cate = cate_hist[-1]
        pos_neg_flag = random.randint(0, 1)
        if pos_neg_flag:
            label, item, cate = 1, pos_item, pos_cate
        else:
            while True:
                neg_item = random.randint(1, len(item_df))
                if neg_item != pos_item:
                    neg_cate = item_df.get_group(neg_item)['cid'].tolist()[0]
                    break
            label, item, cate = 0, neg_item, neg_cate
        
                
        flag = train_valid_test_flag[uid]
        item_hist.pop(-1)
        cate_hist.pop(-1)
        item_hist = '^'.join(map(str, item_hist))
        cate_hist = '^'.join(map(str, cate_hist))
        if flag == 1:
            train_label_array.append(label)
            train_uid_array.append(uid)
            train_iid_array.append(item)
            train_cate_array.append(cate)
            train_hist_iid_array.append(item_hist)
            train_hist_cate_array.append(cate_hist)
            
        elif flag == 2:
            valid_label_array.append(label)
            valid_uid_array.append(uid)
            valid_iid_array.append(item)
            valid_cate_array.append(cate)
            valid_hist_iid_array.append(item_hist)
            valid_hist_cate_array.append(cate_hist)

        elif flag == 3:
            test_label_array.append(label)
            test_uid_array.append(uid)
            test_iid_array.append(item)
            test_cate_array.append(cate)
            test_hist_iid_array.append(item_hist)
            test_hist_cate_array.append(cate_hist)

    train_data = {'label': train_label_array, 'user_id': train_uid_array,
                  'item_id': train_iid_array, 'cate_id': train_cate_array,
                  'item_history': train_hist_iid_array, 'cate_history': train_hist_cate_array}

    valid_data = {'label': valid_label_array, 'user_id': valid_uid_array,
                  'item_id': valid_iid_array, 'cate_id': valid_cate_array,
                  'item_history': valid_hist_iid_array, 'cate_history': valid_hist_cate_array}

    test_data = {'label': test_label_array, 'user_id': test_uid_array,
                 'item_id': test_iid_array, 'cate_id': test_cate_array,
                 'item_history': test_hist_iid_array, 'cate_history': test_hist_cate_array}

    train_df = pd.DataFrame(train_data)
    valid_df = pd.DataFrame(valid_data)
    test_df = pd.DataFrame(test_data)
    train_df.to_csv('train.csv', index=False)
    valid_df.to_csv('valid.csv', index=False)
    test_df.to_csv('test.csv', index=False)
    print(f"The number of users (behavior >= 3):{cnt}")
    print("train, valid, test sample completed")


if __name__ == "__main__":
    max_len, min_len = 300, 3
    # 0.5 neg 0.5 pos
    data_path = "ijcai2016_taobao.csv"
    df = to_df(data_path)
    df, user_len, item_len, cate_len = remap(df, max_len)
    user_df, item_df = gen_user_item_group(df)
    gen_dataset(user_df, item_df, max_len, min_len)
