from flask import Flask, escape, request, render_template, Response, jsonify, json, redirect, url_for, flash

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html');

@app.route('/dataset')
def dataset():
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(True)
    attributes = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
                  'concavity_mean', 'concave points_mean',
                  'symmetry_mean', 'fractal_demension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
                  'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
                  'symmetry_se', 'fractal_demension_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
                  'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst',
                  'symmetry_worst', 'fractal demension_worst']
    return render_template('dataset2.html', X=X, y=y, attr= attributes);


@app.route('/about')
def about():
    return render_template('about.html');

@app.route('/split')
def split():
    from sklearn.model_selection import train_test_split  # bqgi
    from sklearn.datasets import load_breast_cancer  # dataset

    X, y = load_breast_cancer(True)
    attributez = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
                  'concavity_mean', 'concave points_mean',
                  'symmetry_mean', 'fractal_demension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
                  'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
                  'symmetry_se', 'fractal_demension_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
                  'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst',
                  'symmetry_worst', 'fractal demension_worst']
    attributes = ['RM', 'TM', 'PM', 'AM', 'SM', 'CM',
                  'CNM', 'CPM',
                  'SYM', 'FRAC', 'RAD_SE', 'TEX_SE', 'PER_SE', 'AR_SE',
                  'SM_SE', 'COM_SE', 'CON_SE', 'CP_SE',
                  'SYM_SE', 'FRAC_SE', 'RAD_WO', 'TEX_WO', 'PER_WO',
                  'AR_WO', 'SMO_WO', 'COM_WO', 'CP_WO', 'CONP_WO',
                  'SYM_WO', 'FRACD_WO']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)  # sampling


    return render_template('split.html', X_train=enumerate(X_train), X_test=enumerate(X_test), y_train=y_train, y_test=y_test, attr=attributes);

@app.route('/dt')
def dt():
    from sklearn import tree
    from sklearn.model_selection import train_test_split #bqgi
    from sklearn.datasets import load_breast_cancer #dataset

    attributes = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
                  'concavity_mean', 'concave points_mean',
                  'symmetry_mean', 'fractal_demension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
                  'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
                  'symmetry_se', 'fractal_demension_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
                  'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst',
                  'symmetry_worst', 'fractal demension_worst']

    X, y = load_breast_cancer(True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15) #sampling
    classifier = tree.DecisionTreeClassifier(random_state=7)
    classifier.fit(X_train, y_train)  #training
    var = classifier.feature_importances_
    akurasi = (classifier.predict(X_test) == y_test).mean() #akurasi
    return render_template('dt.html', akurasi=(akurasi*100), atribut=30, importances=enumerate(var), names=attributes);

@app.route('/pso')
def pso():
    import pyswarms as ps
    from sklearn import tree
    from sklearn.model_selection import train_test_split
    import numpy as np
    from sklearn.datasets import load_breast_cancer

    attributes = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
                  'concavity_mean', 'concave points_mean',
                  'symmetry_mean', 'fractal_demension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
                  'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
                  'symmetry_se', 'fractal_demension_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
                  'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst',
                  'symmetry_worst', 'fractal demension_worst']

    X, y = load_breast_cancer(True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)
    classifier = tree.DecisionTreeClassifier(random_state=7)
    classifier.fit(X_train, y_train)

    # Define objective function
    def f_per_particle(m, alpha):

        total_features = 30
        # Get the subset of the features from the binary mask
        if np.count_nonzero(m) == 0:
            X_subset = X_train
            X_subset_test = X_test
        else:
            X_subset = X_train[:, m == 1]
            X_subset_test = X_test[:, m == 1]
        # Perform classification and store performance in P
        classifier.fit(X_subset, y_train)
        P = (classifier.predict(X_subset_test) == y_test).mean()

        # Compute for the objective function
        # j = (alpha * (1.0 - P)
        #      + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))

        j = 1 - P

        return j

    def f(x, alpha=0.88):

        n_particles = x.shape[0]
        j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
        return np.array(j)

    # Initialize swarm, arbitrary
    options = {'c1': 0.5, 'c2': 0.5, 'w': 0.9, 'k': 30, 'p': 2}

    # Call instance of PSO
    dimensions = 30  # jumlah feature
    # optimizer.reset()
    optimizer = ps.discrete.BinaryPSO(n_particles=30, dimensions=dimensions, options=options)

    # Perform optimization
    cost, pos = optimizer.optimize(f, iters=100)

    cost_his = optimizer.cost_history
    pos_his = optimizer.pos_history

    # Get the selected features from the final positions
    X_selected_features = X_train[:, pos == 1]  # subset
    X_test_selected_features = X_test[:, pos == 1]  # subset

    # Perform classification and store performance in P
    classifier.fit(X_selected_features, y_train)

    # Compute performance
    subset_performance = (classifier.predict(X_test_selected_features) == y_test).mean()

    return render_template('pso.html', acc=subset_performance, fitur=pos, cost=cost, jml=X_test_selected_features.shape[1], cost_his = enumerate(cost_his[:-1]), pos_his = pos_his[:-1], attr=enumerate(attributes));