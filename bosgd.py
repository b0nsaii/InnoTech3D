import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def SGDClassify(voxel_size=None):
    all_points = []
    all_boxes = []
    
    # Prepare the classifier with warm_start=True to support incremental training
    clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)

    for project in ['Project1', 'Project2', 'Project3', 'Project4']:
        # Load and preprocess data
        s_points = pd.read_csv(f"data/TrainingSet/{project}/{project}.xyz", header=None, names=['x', 'y', 'z', 'i', 'r', 'g', 'b'])
        s_points['project'] = project
        all_points.append(s_points)
        
        s_boxes = pd.read_csv(f'data/TrainingSet/{project}/{project}.csv')
        s_boxes.columns = s_boxes.columns.str.strip()
        s_boxes['project'] = project
        all_boxes.append(s_boxes)
        print(f'The number of boxes in {project} is {len(s_boxes)}')

    # Combine data from all projects
    points = pd.concat(all_points, ignore_index=True)
    boxes = pd.concat(all_boxes, ignore_index=True)
    print(f'Total boxes combined: {len(boxes)}')

    # Data cleaning and preparation
    points[['x', 'y', 'z', 'i', 'r', 'g', 'b']] = points[['x', 'y', 'z', 'i', 'r', 'g', 'b']].apply(pd.to_numeric, errors='coerce')
    boxes[['BB.Min.X', 'BB.Min.Y', 'BB.Min.Z', 'BB.Max.X', 'BB.Max.Y', 'BB.Max.Z']] = boxes[['BB.Min.X', 'BB.Min.Y', 'BB.Min.Z', 'BB.Max.X', 'BB.Max.Y', 'BB.Max.Z']].apply(pd.to_numeric, errors='coerce')
    points = points.dropna()
    boxes = boxes.dropna()

    # Extract features for each bounding box
    point_features = boxes.apply(lambda row: extract_point_features(row, points), axis=1)
    
    # Add bounding box geometric features
    boxes['length'] = boxes['BB.Max.X'] - boxes['BB.Min.X']
    boxes['width'] = boxes['BB.Max.Y'] - boxes['BB.Min.Y']
    boxes['height'] = boxes['BB.Max.Z'] - boxes['BB.Min.Z']
    combined_features = pd.concat([boxes[['length', 'width', 'height']], point_features], axis=1)
    combined_features.columns = combined_features.columns.astype(str)

    X = combined_features
    y = boxes['Label']

    # Train and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Incremental training
    for i in range(0, len(X_train), 100):  # Process in chunks of 100 rows
        clf.partial_fit(X_train.iloc[i:i+100], y_train.iloc[i:i+100], classes=np.unique(y))

    # Test the classifier on the test set
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    return

def extract_point_features(box, points):
    # Mask points within the bounding box
    mask = (
        (points['x'] >= box['BB.Min.X']) & (points['x'] <= box['BB.Max.X']) &
        (points['y'] >= box['BB.Min.Y']) & (points['y'] <= box['BB.Max.Y']) &
        (points['z'] >= box['BB.Min.Z']) & (points['z'] <= box['BB.Max.Z'])
    )
    selected_points = points[mask]

    if selected_points.empty:
        return pd.Series([0] * 10)

    feature_vector = [
        len(selected_points),
        *selected_points[['x', 'y', 'z']].mean().values,
        *selected_points[['x', 'y', 'z']].std().values,
        selected_points['i'].mean(),
        *selected_points[['r', 'g', 'b']].mean().values
    ]
    return pd.Series(feature_vector)

if _name_ == "_main_":
    SGDClassify()