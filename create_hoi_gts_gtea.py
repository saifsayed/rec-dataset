import glob
import pickle
import numpy as np


def get_mean_pos(anchor_bboxes):
    mids = np.zeros((len(anchor_bboxes),2))
    for i, [t_x, t_y, b_x, b_y] in enumerate(anchor_bboxes):
        height = b_y - t_y
        width = b_x - t_x
        if height == 0 or width == 0:
            continue
        mid_x = t_x + width/2
        mid_y = t_y + height/2
        mids[i,:] = [mid_x, mid_y]
    mask = np.logical_not(np.isin(mids, [0,0])[:,0])
    mids_new = mids[mask]
    mids_mean = np.mean(mids_new, axis=0)
    return mids_mean

# def extend_boundary(box_data, start, stop, search_prev, search_next):
def extend_boundary(box_data, start, stop):


    anchor_bboxes = box_data[start:stop]
    anchor_mids = get_mean_pos(anchor_bboxes)

    final_stop = stop
    final_start = start

    start_iter = start
    while start_iter > 0:
        jump_window_mean = get_mean_pos(box_data[start_iter - time_jump_window:start_iter])
        if np.abs(jump_window_mean[0] - anchor_mids[0]) > threshold or np.abs(
                jump_window_mean[1] - anchor_mids[1]) > threshold:
            final_start = start_iter
            break
        start_iter -= time_jump_window

    stop_iter = stop
    while stop_iter < len(box_data):
        jump_window_mean = get_mean_pos(box_data[stop_iter:stop_iter + time_jump_window])
        if np.abs(jump_window_mean[0] - anchor_mids[0]) > threshold or np.abs(
                jump_window_mean[1] - anchor_mids[1]) > threshold:
            final_stop = stop_iter
            break
        stop_iter += time_jump_window

    return final_start, final_stop

def read_gt(data_folder,vid_name, action_name_to_id):
    gt = []
    with open(data_folder + '/gtea/groundTruth/{}.txt'.format(vid_name), 'r') as in_f:
        for x in in_f:
            gt.append(action_name_to_id[x.rstrip()])
    return gt


####Code for extending the boundaries using the author's annotations and then extending the boundaries by tracking the bbobx
time_jump_window = 75 #time threshold for jumping
threshold = 10 #spatial threshold of pixel
data_folder = './data'


action_id_to_name = dict()
action_name_to_id = dict()
with open(data_folder +'/gtea/mapping.txt') as f:
    the_mapping = [tuple(x.strip().split()) for x in f]
    for (i, l) in the_mapping:
        action_id_to_name[int(i)] = l
        action_name_to_id[l] = int(i)

hoi_result_folder = data_folder + '/gtea/bbox_obj_interact'
error_ctr = 0
acc_all = []
timestamp_gts = np.load(data_folder + '/gtea/gtea_annotation_all.npy',allow_pickle=True).item()
timestamp_extended = dict()
timestamp_gt_out = dict()
for vid_name in timestamp_gts:
    print(vid_name)
    vid_name = vid_name.split('.')[0]
    timestamp_gt = np.asarray(timestamp_gts[vid_name+'.txt'])

    with open('{}/{}.pkl'.format(hoi_result_folder,vid_name), 'rb') as in_f:
        hoi_vals = pickle.load(in_f)

    framelevel_gt = read_gt(data_folder,vid_name, action_name_to_id)

    box_data = np.zeros((len(hoi_vals), 4))

    for iter in range(len(hoi_vals)):
        if len(hoi_vals[iter][1]) == 0:
            hoi_vals[iter][1] = np.asarray([0, 0, 0, 0])
        box_data[iter] = hoi_vals[iter][1][0]
    out_labels = np.zeros(len(framelevel_gt), dtype=np.int) - 100
    for frame_idx in timestamp_gt:
        frame_idx = int(frame_idx)
        box_mean = get_mean_pos(box_data[frame_idx-10:frame_idx+10])
        new_start, new_end = extend_boundary(box_data, frame_idx-10, frame_idx+10)
        if new_end > len(framelevel_gt):
            new_end = len(framelevel_gt)-1
        if new_start < 0:
            new_start = 0

        box_mean_extended = get_mean_pos(box_data[new_start:new_end])
        act_val = framelevel_gt[frame_idx]
        gt = action_id_to_name[act_val]
        preds = np.zeros(new_end-new_start) + act_val
        out_labels[new_start:new_end] = act_val
        gts = framelevel_gt[new_start:new_end]

        print(frame_idx, new_start, new_end, len(framelevel_gt),len(preds), len(gts))

        temp = np.sum(np.equal(preds, gts))
        acc = np.sum(np.equal(preds, gts))/len(preds)
        acc_all.append(acc)
        if acc != 1:
            error_ctr += 1
            print('Vidname: {}, Acc: {}, GT: {}, Anchor Frame: {}, New Start: {}, New End: {}, Set GT:{}'.format(vid_name, acc, gt, frame_idx, new_start, new_end, [action_id_to_name[x] for x in set(gts)]))
        timestamp_extended[vid_name] = out_labels
        timestamp_gt_out[vid_name+'.txt'] = timestamp_gt
np.save(data_folder + '/gtea/groundTruth/gtea_annotation_extended_using_authgt_{}_{}.npy'.format(time_jump_window,threshold), timestamp_extended)
np.save(data_folder + '/gtea/groundTruth/gtea_annotation_using_authgt_{}_{}.npy'.format(time_jump_window,threshold), timestamp_gt_out)
