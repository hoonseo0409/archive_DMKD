import pandas as pd
from datetime import datetime
from copy import deepcopy
# from skimage.feature import hog
import mahotas
import imageio
import os
import numpy as np
from math import floor
import json
import plotly.graph_objects as go

def convert_time_to_minutes(time_str):
    if time_str == "no image":
        return time_str
    time_str = time_str[1:]
    hours = int(time_str.split("h")[0])
    minutes = int((time_str.split("h")[1]).split("m")[0])
    date = (time_str.split("m")[1]).split("_")[0]
    date_format = "%Y-%m-%d"
    days = (datetime.strptime(date, date_format) - datetime.strptime('1022-11-21', date_format)).days

    return minutes + 60 * hours + 60 * 24 * days

def load_austin(path_to_austin_image_info, path_to_images, path_to_satellite_images, prediction_interval= 60 * 5, num_records = 5, num_bags = 10, label_columns= None, model = "pftas", model_kwargs = None, img_split = None, dynamic_features = None):
    if label_columns is None: label_columns = ["NORTH_THRU", "NORTH_RIGHT", "NORTH_LEFT", "SOUTH_THRU", "SOUTH_RIGHT", "SOUTH_LEFT", "EAST_THRU", "EAST_RIGHT", "EAST_LEFT", "WEST_THRU", "WEST_RIGHT", "WEST_LEFT", "UNASSIGNED_THRU", "UNASSIGNED_RIGHT", "UNASSIGNED_LEFT"]
    if img_split is None: img_split = [3, 2]
    # static_features = ["Road_Class_WE", "Road_Class_NS", "One_Way_WE", "One_Way_NS", "Zoning_Code_NW", "Zoning_Type_NW", "Zoning_Code_NE", "Zoning_Type_NE", "Zoning_Code_SW", "Zoning_Type_SW", "Zoning_Code_SE", "Zoning_Type_SE", "FLUM", "Urban", "Proximity_to_Wildland"]
    ## Set default model params
    if model == "hog":
        ## Num features of HOG = orientations * , approximately proportional to cells_per_block and 1 / pixels_per_cell, 
        ## Exact features of HOG = int((width / pixelspercell[0]) - (cellsperblock[0] -1)) * int((width / pixelspercell[1]) - (cellsperblock[1] -1)) * (cellsperblock[0] * cellsperblock[1]) * orientations
        if model_kwargs is None: model_kwargs = dict(orientations= 4, pixels_per_cell=(12, 12), cells_per_block=(2, 2), visualize= True, multichannel=True)
        image_converter = lambda img: hog(img, **model_kwargs)
    elif model == "pftas":
        if model_kwargs is None: model_kwargs = dict()
        image_converter = lambda img: mahotas.features.pftas(img, **model_kwargs)
    else:
        raise Exception(f"Unsupported Model: {model}")
    if dynamic_features is None: dynamic_features = ["minutes_scaled", "Speed_Limit_WE"]

    static_data = [] ## (num_participants, dim_static)
    static_features_encoded = []
    dynamic_data = {'image_0': [], 'image_5': [], 'image_10': [], "misc": []} ## {direction: (num_participants, dim_dynamic, num_records)}
    dynamic_features_region = {'image_0': {"regions": [], "idc": []}, 'image_5': {"regions": [], "idc": []}, 'image_10': {"regions": [], "idc": []}, "misc": {"regions": dynamic_features, "idc": [0, 1]}} ## {direction: (num_participants, dim, num_records)}
    dynamic_imgs_path = {'image_0': [], 'image_5': [], 'image_10': []}
    target_labels = {"name_raw": label_columns, "name_readable": label_columns, "data_arr": []} ## data_arr: (num_participants, num_targets, num_records)
    img_features_range = {"max": - 1e+8, "min": 1e+8}
    features_range = {}
    similarities = []

    mins_tick = 60
    speeds_trends_of_week = {day: [[[] for j in range(60 // mins_tick)] for i in range(24)] for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]}

    if path_to_satellite_images.endswith("/"): path_to_satellite_images= path_to_satellite_images[:-1]
    if path_to_images.endswith("/"): path_to_images= path_to_images[:-1]
    dataset_processed_path = f"./out/traffic_{model}_{img_split[0]}-{img_split[1]}"
    if not os.path.exists(dataset_processed_path): os.mkdir(dataset_processed_path)
    for direction in ["image_0", "image_5", "image_10"]:
        if not os.path.exists(f"{dataset_processed_path}/{direction}"): os.mkdir(f"{dataset_processed_path}/{direction}")
    
    locationid_to_satellite = {}
    for filename in os.listdir(path_to_satellite_images):
        if filename.endswith(".jpg"):
            img_features = np.array(image_converter(imageio.imread(f'{path_to_satellite_images}/{filename}')))
            locationid_to_satellite[filename.split(".")[0]] = img_features
            img_features_range["max"], img_features_range["min"] = max(img_features_range["max"], max(img_features)), min(img_features_range["min"], min(img_features))
    dim_satellite = next(iter(locationid_to_satellite.values())).shape[0]

    austin_image_info_df = pd.read_csv(path_to_austin_image_info)
    # austin_image_info_df = austin_image_info_df.drop((austin_image_info_df[austin_image_info_df.image_0 == "no image" or austin_image_info_df.image_5 == "no image" or austin_image_info_df.image_10 == "no image"]).index)
    dynamic_filenames = os.listdir(path_to_images)
    austin_image_info_df = austin_image_info_df.drop(austin_image_info_df[(austin_image_info_df['image_0'] == "no image") | (austin_image_info_df['image_5'] == "no image") | (austin_image_info_df['image_10'] == "no image") | (~austin_image_info_df['image_0'].isin(dynamic_filenames)) | (~austin_image_info_df['image_5'].isin(dynamic_filenames)) | (~austin_image_info_df['image_10'].isin(dynamic_filenames))].index)
    austin_image_info_df = austin_image_info_df.fillna(0)

    austin_image_info_df['minutes'] = austin_image_info_df.apply(lambda row: convert_time_to_minutes(row.image_0), axis = 1)
    austin_image_info_df['minutes_scaled'] = austin_image_info_df['minutes']

    ## interpolates the missing labels.
    austin_image_info_df["datetime"] = pd.to_datetime(austin_image_info_df['time_id'], format = "%H:%M %Y-%m-%d") ## not '2022/11/21  8:15:00 PM', yes '09:00 2022-11-22'
    austin_image_info_df = austin_image_info_df.set_index("datetime")
    austin_image_info_df_labels = austin_image_info_df[label_columns]
    # # set_index(austin_image_info_df['minutes_scaled'])
    austin_image_info_df_labels = austin_image_info_df_labels.replace(to_replace= -1, value= np.nan)
    austin_image_info_df_labels = austin_image_info_df_labels.interpolate(method= "time")
    austin_image_info_df_labels = austin_image_info_df_labels.fillna(method = 'bfill')
    austin_image_info_df[label_columns] = austin_image_info_df_labels[label_columns]

    dummy_to_categ_map = {}
    for categ_feature in ["Road_Class_WE", "Road_Class_NS", "One_Way_WE", "One_Way_NS", "Zoning_Code_NW", "Zoning_Type_NW", "Zoning_Code_NE", "Zoning_Type_NE", "Zoning_Code_SW", "Zoning_Type_SW", "Zoning_Code_SE", "Zoning_Type_SE", "FLUM", "Urban", "Proximity_to_Wildland"]:
        df_dummy = austin_image_info_df.loc[:, categ_feature].str.get_dummies()
        dummy_col_names = [categ_feature + '_is_' + col for col in df_dummy.columns]
        for dummy in dummy_col_names:
            dummy_to_categ_map[dummy] = categ_feature
        static_features_encoded += dummy_col_names
        df_dummy.columns = deepcopy(dummy_col_names)
        austin_image_info_df = pd.concat([austin_image_info_df, df_dummy], axis=1)
    static_features_encoded += ["Speed_Limit_WE", "Speed_Limit_NS"] ## numerical static features

    for numer_feature in ["Latitude", "Longitude", "Speed_Limit_WE", "Speed_Limit_NS", "minutes_scaled"] + label_columns:
        austin_image_info_df[numer_feature + "_raw"] = austin_image_info_df[numer_feature]
        numer_feature_column = austin_image_info_df[numer_feature]
        features_range[numer_feature] = {}
        features_range[numer_feature]["max-min"], features_range[numer_feature]["min"] = max(numer_feature_column.max() - numer_feature_column.min(), 1e-8), numer_feature_column.min()
        austin_image_info_df[numer_feature] = (numer_feature_column - features_range[numer_feature]["min"]) / features_range[numer_feature]["max-min"]
    
    # with open("./WEST_THRU.json", 'w') as f:
    #     json.dump(austin_image_info_df["WEST_THRU"].to_list(), f, indent=2) 
    
    locationid_to_size = dict(austin_image_info_df.groupby('location_id').size())
    locationid_to_sample_idx = {key: 0 for key in locationid_to_size.keys()}
    if False:
        import plotly.graph_objects as go
        hours = range(9, 19)
        days = ["Tuesday", "Wednesday", "Thursday", "Friday"]
        label_columns_raw = [name + "_raw" for name in label_columns]
        x = list(range(1, len(hours) * (60 // mins_tick) + 1))
        y = {day: [] for day in days}
        plot_data = []
        for index, row in austin_image_info_df.iterrows():
            time_id = row["time_id"]
            day, hour, minute = datetime.strptime(time_id.split(" ")[1], "%Y-%m-%d").strftime('%A'), int(time_id.split(":")[0]), int(time_id.split(":")[1].split(" ")[0])
            speeds_trends_of_week[day][hour][minute // mins_tick].append(np.mean(row[label_columns_raw]))
        speeds = 0.0
        for day in days:
            for hour in hours:
                for div in range(60 // mins_tick):
                    if len(speeds_trends_of_week[day][hour][div]) > 0:
                        speeds = speeds_trends_of_week[day][hour][div]
                    else:
                        print(f"Empty speeds, {day}, {hour}, {div}.")
                    y[day].append(np.mean(speeds))
            plot_data.append(go.Scatter(x= x, y= y[day], name= day))
        ## to avoid zigzag line.
        compress = 3
        xnew = []
        ynew = []
        # for idx in range(len(x)):

        fig = go.Figure(plot_data)

        fig.update_layout(
            xaxis = dict(
                tickmode = 'array',
                tickvals = [1 + i * (60 // mins_tick) for i in range(len(hours))],
                ticktext = list(hours),
                tickfont = dict(size=35),
                # zeroline=True,
                showline=True, linewidth=2, linecolor='black', mirror=True, ticks= "inside", tickwidth= 5, ticklen= 15,
            ),
            yaxis = dict(tickfont = dict(size=35), 
            # zeroline=True,
            showline=True, linewidth=2, linecolor='black', mirror=True, ticks= "inside", tickwidth= 5, ticklen= 15,), # range= [12.0, 17.5]
            # paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )

        # fig.show()
        fig.write_html(f"{dataset_processed_path}/speeds_trends.html")


    austin_image_info_df = austin_image_info_df.sort_values(['location_id', 'minutes'], ascending=True).groupby('location_id')
    # austin_image_info_df = austin_image_info_df.groupby(by= "location_id")
    current_num_bags = 0
    while(current_num_bags < num_bags):
        if num_bags > 20 and current_num_bags % int(num_bags / 20) == 0:
            print(f"{current_num_bags} bags are processed.")
        if all([value == -1 for value in locationid_to_sample_idx.values()]):
            print(f"Requested number of bags is {num_bags} but it exceeds current maximum bags {current_num_bags}, so we end here.")
            break
        for locationid, df_group in austin_image_info_df:
            if locationid_to_sample_idx[locationid] != -1:
                if locationid_to_sample_idx[locationid] + num_records > locationid_to_size[locationid]:
                    locationid_to_sample_idx[locationid] = -1
                    break
                else:
                    prediction_target_idx = None
                    for idx in range(locationid_to_sample_idx[locationid] + num_records, locationid_to_size[locationid]):
                        if df_group.iloc[idx]['minutes'] > df_group.iloc[locationid_to_sample_idx[locationid] + num_records]['minutes'] + prediction_interval:
                            prediction_target_idx = idx
                    if prediction_target_idx is None:
                        locationid_to_sample_idx[locationid] = -1
                        break
                    # target_labels["data_arr"].append([[df_group.iloc[prediction_target_idx]['minutes']]])            
                static_data.append(np.concatenate((df_group.iloc[0][static_features_encoded].to_numpy(), locationid_to_satellite[str(int(locationid))]), axis= 0))
                similarities.append([])
                
                records_indices = list(range(locationid_to_sample_idx[locationid], locationid_to_sample_idx[locationid] + num_records)) + [prediction_target_idx]
                for direction in ["image_0", "image_5", "image_10"]:
                    dynamic_data[direction].append([])
                    dynamic_features_region[direction]["regions"].append([])
                    dynamic_features_region[direction]["idc"].append([])
                    dynamic_imgs_path[direction].append([])
                    for record_idx in records_indices:
                        if os.path.exists(f"{dataset_processed_path}/{direction}/{record_idx}_idc.json"):
                            for target, filename in zip([dynamic_data[direction][-1], dynamic_features_region[direction]["regions"][-1], dynamic_features_region[direction]["idc"][-1]], ["data", "regions", "idc"]):
                                with open(f"{dataset_processed_path}/{direction}/{record_idx}_{filename}.json", 'r') as f:
                                    target.append(json.load(f))
                        else:
                            dynamic_data[direction][-1].append([])
                            dynamic_features_region[direction]["regions"][-1].append([])
                            dynamic_features_region[direction]["idc"][-1].append([])
                            img_curr = imageio.imread(f"{path_to_images}/{df_group.iloc[record_idx][direction]}")
                            image_width, image_height = img_curr.shape[1], img_curr.shape[0]
                            patch_width, patch_height = floor(image_width / img_split[0]), floor(image_height / img_split[1])
                            for x_idx in range(0, img_split[0]):
                                for y_idx in range(0, img_split[1]):
                                    patch_length_previous = len(dynamic_data[direction][-1][-1])
                                    # dynamic_features_region[direction]["regions"][-1][-1].append([x_idx * patch_width, (x_idx + 1) * patch_width, y_idx * patch_height, (y_idx + 1) * patch_height])
                                    dynamic_features_region[direction]["regions"][-1][-1].append([x_idx, y_idx])

                                    dynamic_data[direction][-1][-1] += list(image_converter(img_curr[y_idx * patch_height : (y_idx + 1) * patch_height, x_idx * patch_width : (x_idx + 1) * patch_width]))
                                    dynamic_features_region[direction]["idc"][-1][-1].append([patch_length_previous, len(dynamic_data[direction][-1][-1])])
                            for data, filename in zip([dynamic_data[direction][-1][-1], dynamic_features_region[direction]["regions"][-1][-1], dynamic_features_region[direction]["idc"][-1][-1]], ["data", "regions", "idc"]):
                                with open(f"{dataset_processed_path}/{direction}/{record_idx}_{filename}.json", 'w') as f:
                                    json.dump(data, f, indent=2)
                        dynamic_imgs_path[direction][-1].append(f"{path_to_images}/{df_group.iloc[record_idx][direction]}")
                
                dynamic_data["misc"].append([])
                target_labels["data_arr"].append([])
                for record_idx in records_indices:
                    dynamic_data["misc"][-1].append(df_group.iloc[record_idx][dynamic_features].to_list())
                    target_labels["data_arr"][-1].append(df_group.iloc[record_idx][label_columns].to_list())
                for i in records_indices:
                    similarities[-1].append([])
                    for j in records_indices:
                        similarities[-1][-1].append(1. / max(df_group.iloc[i]["minutes_scaled"] - df_group.iloc[j]["minutes_scaled"], 1e-8))
                locationid_to_sample_idx[locationid] += 1
                current_num_bags += 1
                if current_num_bags >= num_bags: break
    static_data = np.array(static_data)
    static_features_encoded += ["satellite" for i in range(dim_satellite)]
    for modality in ["image_0", "image_5", "image_10", "misc"]:
        dynamic_data[modality] = np.transpose(np.array(dynamic_data[modality]), (0, 2, 1))
    target_labels["data_arr"] = np.transpose(np.array(target_labels["data_arr"]), (0, 2, 1))

    return static_data, static_features_encoded, dynamic_data, dynamic_features_region, dynamic_imgs_path, target_labels, similarities, features_range

