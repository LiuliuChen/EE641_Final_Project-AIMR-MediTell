"""
EDA and preprocessing of Indiana X-Rays images
Split train, validation and test dataset and save them
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_data(projection_path, reports_path):
    """
    We only consider the frontal X-ray image
    """
    df_projections = pd.read_csv(projection_path)
    df_report = pd.read_csv(reports_path)

    print("Number of Unique patients: ", len(df_projections['uid'].unique()))

    # find valid uid
    valid_ids = []
    count_list = df_projections.groupby(by='uid').nunique()
    for i in count_list.index:
        # choose uid with two projection label
        if count_list.loc[i, 'projection'] == 2:
            valid_ids.append(i)

    # select only frontal images and find their corresponding image filename
    user_proj_cnt = {}
    valid_df = pd.DataFrame(columns=['uid', 'filename', 'projection'])

    idx_cnt = 0
    for i in valid_ids:
        front_lat_cnt = {'Frontal': 0}
        user_proj_cnt[i] = front_lat_cnt

        temp = df_projections[df_projections['uid'] == i]
        for row in temp.iterrows():
            projection = row[-1]['projection']
            if projection == 'Frontal':
                if front_lat_cnt['Frontal'] == 0:
                    front_lat_cnt['Frontal'] += 1
                    valid_df.loc[idx_cnt, :] = row[-1]
                else:
                    continue

            idx_cnt += 1

    valid_df = valid_df.drop_duplicates()

    # find corresponding diagnostic reports for images
    dataset = pd.merge(valid_df, df_report, on=['uid'])
    print('Final dataset shape', dataset.shape)

    return dataset


# split train, validation and test dataset, and save them
def split_train_val_test_info(dataset, dataset_saved_path):
    dataset = dataset.dropna(subset=['findings'])

    train, test = train_test_split(dataset, test_size=0.4)
    test, val = train_test_split(train, test_size=0.5)

    train.to_csv(f'{dataset_saved_path}Final_Train_Data.csv', index=False)
    val.to_csv(f'{dataset_saved_path}Final_CV_Data.csv', index=False)
    test.to_csv(f'{dataset_saved_path}Final_Test_Data.csv', index=False)


def main():
    projection_path = '../../autodl-tmp/medical_projects/archive/indiana_projections.csv'
    report_path = '../../autodl-tmp/medical_projects/archive/indiana_reports.csv'
    dataset_saved_path = '../../autodl-tmp/medical_projects/archive/'

    # preprocess data
    dataset = preprocess_data(projection_path, report_path)
    dataset.to_csv(f'{dataset_saved_path}indiana_images_info.csv')

    # split train, val and test dataset, and save them
    split_train_val_test_info(dataset, dataset_saved_path)


if __name__ == "__main__":
    main()


