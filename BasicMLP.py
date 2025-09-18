from sklearn.neural_network import MLPClassifier

def BasicMLP(hidden_layer_size=8, learning_rate_init=0.01, max_iter=5000):

    return MLPClassifier(
            hidden_layer_sizes=(hidden_layer_size,),
            max_iter=max_iter,
            learning_rate_init=learning_rate_init,
            activation="tanh",
            solver="adam",
            alpha=0.0,
            batch_size="auto",            
            learning_rate="constant",
            power_t=0.5,
            shuffle=True,
            random_state=589,
            tol=1e-10,
            verbose=False,
            warm_start=False,
            momentum=0.0,
            nesterovs_momentum=False,
            early_stopping=False,            
            validation_fraction=0.1,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            n_iter_no_change=100,
            max_fun=1000000,
        )
