import numpy as np
import copy
from tqdm import tqdm
AGENT_ATTR = 0
AGENT_ID = 0
ACTOR_ATTR = 1
LANE_ATTR = 2

def padding(dict_):
    #살아남은 요소들은 (x,y)에 빈공간에 0을 채워준다. 동시에 concat할 패딩도 만들어준다
    for key, val in dict_.items():
        length = 50
        padding = np.zeros(50)
        new_x = np.zeros(50)
        new_y = np.zeros(50)
        timestamp = val['timestamp']
        x = val['x']
        y = val['y']
        i = 0
        for idx in range(length):
            if timestamp[i] == idx+1:
                padding[idx] = 1
                new_x[idx] = x[i]
                new_y[idx] = y[i]
                i +=1
                if len(timestamp) == i:
                    break
        dict_[key]['x'] = new_x
        dict_[key]['y'] = new_y
        dict_[key]['padding'] = padding

def get_local_lanes(agent, df, avm):
    city_name = df["CITY_NAME"].values[0]
    seq_lane_props = avm.city_lane_centerlines_dict[city_name]
    lane_dict = {}
    x_offset = 50
    y_offset = 50
    x_min = int(max(min(agent["x"]) - x_offset/2,0))
    x_max = int(max(agent["x"]) + x_offset/2)
    y_min = int(max(min(agent["y"]) - y_offset/2,0))
    y_max = int(max(agent["y"]) + y_offset/2)
    for lane_id, lane_props in seq_lane_props.items():
        lane_cl = lane_props.centerline
        if np.min(lane_cl[:, 0]) < x_max and np.min(lane_cl[:, 1]) < y_max and np.max(lane_cl[:, 0]) > x_min and np.max(lane_cl[:, 1]) > y_min:
            temp_dict = {}
            temp_dict["centerline"] = lane_cl
            temp_dict["has_traffic_control"] = lane_props.has_traffic_control
            temp_dict["turn_direction"] = lane_props.turn_direction
            temp_dict["is_intersection"] = lane_props.is_intersection
            temp_dict["l_neighbor_id"] = lane_props.l_neighbor_id
            temp_dict["r_neighbor_id"] = lane_props.r_neighbor_id
            temp_dict["predecessors"] = lane_props.predecessors
            temp_dict["successors"] = lane_props.successors
            x1 = np.array(agent["x"]).reshape(-1,1)
            y1 = np.array(agent["y"]).reshape(-1,1)
            xy = np.hstack([x1,y1]).astype(np.float)
            # temp_dict["candidate_centerlines"]= avm.get_candidate_centerlines_for_traj(xy,city_name, max_search_radius=3)
            # 이건 target vehicle이 점유하는 centerline을 잡는데 쓰자.
            # lane id가 안나와서 제한적임.
            # vectornet에는 불필요할듯
            lane_dict[str(lane_id)] = temp_dict
    return lane_dict

def arrange2id(Track_id_list, df):
    dict_ = {}
    for track_id in Track_id_list:
        str_query = "TRACK_ID == @track_id"
        dic = {}
        result = df.query(str_query).to_numpy()
        timestamp = result[:,5]
        x = result[:,2]
        y = result[:,3]
        dic['timestamp'] = timestamp
        dic['x']=x
        dic['y']=y
        dict_[track_id] = dic
    return dict_

def change_timestamp(df):
    sample_track_id = '00000000-0000-0000-0000-000000000000'
    str_expr = "TRACK_ID == @sample_track_id"
    result = df.query(str_expr).to_numpy()
    unique_timestamp = result[:,0]
    unique_timestamp_fixed = []
    for idx in unique_timestamp:
        unique_timestamp_fixed.append(str(idx))
#     x = result[:,3]
#     y = result[:,4]
    time_dict = {}
    time = 1
    for i in unique_timestamp_fixed:
        time_dict[i] = time
        time = time + 1
    time_stamp = df['TIMESTAMP'].to_numpy()
    new_time_stamp = []
    for item in time_stamp:
        new_time_stamp.append(time_dict[str(item)])
    df = df.drop(['TIMESTAMP'],axis=1)
    df['TIMESTAMP'] = new_time_stamp
    return df

def change_timestamp(df):
    sample_track_id = '00000000-0000-0000-0000-000000000000'
    str_expr = "TRACK_ID == @sample_track_id"
    result = df.query(str_expr).to_numpy()
    unique_timestamp = result[:,0]
    unique_timestamp_fixed = []
    for idx in unique_timestamp:
        unique_timestamp_fixed.append(str(idx))
#     x = result[:,3]
#     y = result[:,4]
    time_dict = {}
    time = 1
    for i in unique_timestamp_fixed:
        time_dict[i] = time
        time = time + 1
    time_stamp = df['TIMESTAMP'].to_numpy()
    new_time_stamp = []
    for item in time_stamp:
        new_time_stamp.append(time_dict[str(item)])
    df = df.drop(['TIMESTAMP'],axis=1)
    df['TIMESTAMP'] = new_time_stamp
    return df

def get_agent_track_id(df):
    str_expr = "OBJECT_TYPE == 'AGENT'"
    result = df.query(str_expr).to_numpy()
    return result[0][0]

def remove_lazy(dict_,thres):
    rm_list = []
    for key, val in dict_.items():
        first_x = val['x'][0]
        last_x = val['x'][-1]
        l2_dist = np.sqrt((val['x'][-1] - val['x'][0])**2 + (val['y'][-1] - val['y'][0])**2)
        if l2_dist < thres:
            rm_list.append(key)
    for i in rm_list:
        del dict_[i]

def transforms(_dict):
    # 얕은 복사이므로 매개변수로 받은 것이 변함
    target_x = _dict['agent']['x'][20]
    target_y = _dict['agent']['y'][20]
    _dict['agent']['x'] -= target_x
    _dict['agent']['y'] -= target_y
    for key, actor in _dict['actor'].items():
        mask = actor['padding'].astype('int8')
        array_mask = mask > 0
        actor['x'][array_mask] -= target_x
        actor['y'][array_mask] -= target_y
        
    for key, lane in _dict['lanes'].items():
        lane['centerline'][:,0] -= target_x
        lane['centerline'][:,1] -= target_y        

def connect_lane(_dict):
    _dict['connected_lane'] = []
    for lane_key, lane in _dict['lanes'].items():
        # 이전에 발견된 적 있는지 확인
        is_new = True
        where = -1
        for i, out_list in enumerate(_dict['connected_lane']):
            if lane_key in out_list:
                is_new = False
                where = i
        if is_new:
            temp_list = []
            temp_list.append(lane_key)
            pred = lane['predecessors']
            succ = lane['successors']
            if pred != None:
                pred = str(lane['predecessors'][0])
                if pred in list(_dict['lanes'].keys()):
                    temp_list.insert(0,pred)
            if succ != None:
                succ = str(lane['successors'][0])
                if succ in list(_dict['lanes'].keys()):
                    temp_list.append(succ)
            _dict['connected_lane'].append(temp_list)
        else:
            i = where # 몇 번째 lane list에 있는지 확인함
            #print(lane['predecessors'])
            #lane predecessors = None
            pred = lane['predecessors']
            succ = lane['successors']
            if pred != None:
                pred = str(lane['predecessors'][0])
                if pred in list(_dict['lanes'].keys()) and pred not in _dict['connected_lane'][i]:
                    _dict['connected_lane'][i].insert(0,pred)
            if succ != None:
                succ = str(lane['successors'][0])
                if succ in list(_dict['lanes'].keys()) and succ not in _dict['connected_lane'][i]:
                    _dict['connected_lane'][i].append(succ)
        

def stitching_lane(_dict):
    _dict['stitch_lanes'] = []
    stitch_list = _dict['connected_lane']
    for lane_list in stitch_list:
        arr = None
        for idx, lane in enumerate(lane_list):
            if idx == 0:
                arr = _dict['lanes'][lane]['centerline']
                continue
            else:
                arr = np.vstack((arr, _dict['lanes'][lane]['centerline']))
        _dict['stitch_lanes'].append(arr)

def make_agent_fm(_dict, POLY_ID, past_len):
    # No Padding
    length = past_len
    feature_matrix = np.zeros((length-2, 6))
    x_start = _dict['agent']['x'][:length-2].T
    y_start = _dict['agent']['y'][:length-2].T
    x_end = _dict['agent']['x'][1:length-1].T
    y_end = _dict['agent']['y'][1:length-1].T
    feature_matrix[:,0] = x_start
    feature_matrix[:,1] = y_start
    feature_matrix[:,2] = x_end
    feature_matrix[:,3] = y_end
    feature_matrix[:,4] = AGENT_ATTR
    feature_matrix[:,5] = POLY_ID
    POLY_ID += 1
    _dict['agent_fm'] = feature_matrix
    return POLY_ID

def make_actor_fm(_dict, POLY_ID, past_len):
    # No Padding
    _dict["actor_fm"] = []
    for key, actor in _dict['actor'].items():
        x = actor['x'][:past_len]
        y = actor['y'][:past_len]
        padding = actor['padding'][:past_len]
        mask = padding > 0
        x_nopad = x[mask]
        y_nopad = y[mask]
        length = len(x_nopad)
        if length < 2:
            return POLY_ID
        feature_matrix = np.zeros((length-2, 6))
        x_start = x_nopad[:length-2]
        y_start = y_nopad[:length-2]
        x_end = x_nopad[1:length-1]
        y_end = y_nopad[1:length-1]
        feature_matrix[:,0] = x_start
        feature_matrix[:,1] = y_start
        feature_matrix[:,2] = x_end
        feature_matrix[:,3] = y_end
        feature_matrix[:,4] = AGENT_ATTR
        feature_matrix[:,5] = POLY_ID
        POLY_ID += 1
        _dict["actor_fm"].append(feature_matrix)
    return POLY_ID

def make_lane_fm(_dict, POLY_ID):
    # No Padding
    _dict["lane_fm"] = []
    for lane in _dict['stitch_lanes']:
        x = np.array(lane[:,0])
        y = np.array(lane[:,1])
        length = len(x)
        feature_matrix = np.zeros((length-2, 6))
        x_start = x[:length-2]
        y_start = y[:length-2]
        x_end = x[1:length-1]
        y_end = y[1:length-1]

        feature_matrix[:,0] = x_start
        feature_matrix[:,1] = y_start
        feature_matrix[:,2] = x_end
        feature_matrix[:,3] = y_end
        feature_matrix[:,4] = LANE_ATTR
        feature_matrix[:,5] = POLY_ID
        POLY_ID += 1
        _dict["lane_fm"].append(feature_matrix)
    return POLY_ID

def make_edge_index(_dict):
    # agent
    length = _dict['agent_fm'].shape[0]
    row1 = np.array(range(0,length-2)).astype('int8')
    row2 = np.array(range(1,length-1)).astype('int8')
    agent_edge_matrix = np.vstack([row1, row2])
    # actor
    actor_edge_matrix = []
    for actor in _dict['actor_fm']:
        length = actor.shape[0]
        row1 = np.array(range(0,length-2)).astype('int8')
        row2 = np.array(range(1,length-1)).astype('int8')
        actor_edge_matrix.append(np.vstack([row1, row2]))
    # lane
    lane_edge_matrix = [] 
    for lane in _dict['lane_fm']:
        length = lane.shape[0]
        row1 = np.array(range(0,length-2)).astype('int8')
        row2 = np.array(range(1,length-1)).astype('int8')
        lane_edge_matrix.append(np.vstack([row1, row2]))
                    
    _dict['agent_edge_matrix'] = agent_edge_matrix
    _dict['actor_edge_matrix'] = actor_edge_matrix
    _dict['lane_edge_matrix'] = lane_edge_matrix

def get_trainable_set(root_dir, idx, file, afl, avm):
    temp_dict = {}
    #seq_path = root_dir + file
    #df = afl.get(seq_path).seq_df
    df = afl[idx].seq_df
    df = change_timestamp(df)
    Track_id_list = afl[idx].track_id_list
    dict_ = arrange2id(Track_id_list, df)

    #dict_에서 AV는 필요없으니까 지운다
    av_track_id = '00000000-0000-0000-0000-000000000000'
    av = dict_[av_track_id]
    del dict_[av_track_id]
    #Agent는 따로 빼준다
    agent_track_id = get_agent_track_id(df)
    agent = dict_[agent_track_id]
    del dict_[agent_track_id]

    #Others에서 너무 적게 시작 지점과 마지막 지점을 통해 너무 적게 움직인 agent는 제거한다
    thres = 10
    before_padding_dict = copy.deepcopy(dict_)
    remove_lazy(dict_,thres)
    before_padding_dict = copy.deepcopy(dict_) # visualize를 위해 padding없는 버전 백업
    padding(dict_)
    temp_dict['agent'] = agent
    temp_dict['actor'] = dict_
    temp_dict['actor_NoPadding'] = before_padding_dict
    temp_dict['lanes'] = get_local_lanes(agent, df, avm)
    #temp_dict['drivable_area'] = get_local_drivable_area(agent, df)
    temp_dict['df'] = df
    return temp_dict