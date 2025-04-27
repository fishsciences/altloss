data {
    /* Fish data */
    int<lower=1> N_fish;               // total number of fish in the data
    vector<lower=0>[N_fish] fl_data;   // fork length for each fish in the dataset

    /* 
     ***** Transition structure *****
     *
     * The movement of the fish through the facility is modeled
     * as a Markov renewal process.Such a process satisfies the
     * Markov property only ot the times when state transitions
     * take place, and is described by the transition probability
     * matrix:
     * 
     *   P_ij = P(next state is j | current state is i)
     *
     * and the residence time probability matrix:
     *
     *  F_ij(t) = P(t < residence time <= t + dt | current state i, next state j)
     *
     * The system has N_states states, some of which are absorbing 
     * states (once the system enters such a state, it canot leave 
     * it), and the rest are transient. We assume the first 
     * N_transient states are transient, and the remaining states
     * are absorbing
     *
     * The `transitions` array holds the subset of N_transitions
     * transitions that are possible between the N_states states of 
     * the system. The probabilities of all other transitions will 
     * be pinned at zero. An allowed transitions is represented as
     * a pair of states.
     */
    int<lower=2> N_states;
    int<lower=0, upper=(N_states-2)> N_transient;
    int<lower=1, upper=N_states*N_states> N_transitions;
    array[2, N_transitions] int<lower=1, upper=N_states> transitions;
    
    /* 
     *****  Observation data *****
     *
     * We take the view that the observed data consits of residence 
     * times and state transitions, and both of these components are
     * imperfectly known. 
     *
     * Instead of a residence time we only have a lower bound and an
     * upper bound (t_lower and t_upper)
     *
     * For transitions, we know the current state, but we only know
     * a set of next states the fish could have transitioned to. 
     * These potential next states are encoded as a {0, 1} vector
     * with N_states components.
     */
    int<lower=1> N_obs;
    array[N_obs] int<lower=1, upper=N_states> curr;
    array[N_obs, N_states] int<lower=0, upper=1> next;
    array[N_obs] int<lower=1, upper=N_fish> fish;
    vector<lower=0>[N_obs] t_lower;
    vector<lower=0>[N_obs] t_upper;
    
}

transformed data {
    /* rescaled log fork length */
    real offset_fl = mean(log(fl_data));
    real<lower=0> scale_fl = sd(log(fl_data));
    vector[N_fish] log_fl;
    array[N_obs] row_vector<upper=0>[N_states] log_next;

    /* rescaled time */
    real scale_time = mean(0.5 * (t_upper + t_lower));
    array[2] vector<lower=0>[N_obs] t;
    
    log_fl = (log(fl_data) - offset_fl) / scale_fl;
    t[1] = t_lower / scale_time;
    t[2] = t_upper / scale_time;

    /* validate absorbing states */
    for (k in 1:N_transitions) {
        if (transitions[1, k] > N_transient)
	    reject("State ", transitions[1, k], " is transient, but was declared absorbing");
    }

    /* check for disallowed transitions in the data */
    for (i in 1:N_obs) {
        for (j in 1:N_states) {
	    int allowed = 0;
	    if (next[i][j] <= 0) continue;
            for (k in 1:N_transitions) {
                if ((curr[i] == transitions[1, k]) && (j == transitions[2, k])) {
                    allowed = 1;
                    break;
                }
            }
	    if (!allowed) {
	        reject("disallowed transition found:", curr[i], " -> ", j);
	    }
	}
    }

    for (i in 1:N_obs) {
        log_next[i] = log(to_row_vector(next[i, :]));
    }
}

parameters {

    /*
     * True reesidence time for each observation
     */
    vector<lower=t[1], upper=t[2]>[N_obs] tau;
    
    /* 
     ***** Residence time distribution parameters *****
     *
     * The residence time distributions (elements of 
     * the F_ij matrix) are modelled as Weibull 
     * distributions with means mu[i] and shape
     * parameters kappa[i]. A linear regression model
     * with slope beta[i] and intercept alpha[i] 
     * links the mean residence time to the fish 
     * fork length, on a logarithmic scale.
     */
    vector<offset=1>[N_transitions] alpha_res;
    vector[N_transitions] beta_res;
    vector<lower=0>[N_transitions] kappa_res;
    
    /* 
     ***** EMC transition intensities ***** 
     *
     * 
     */

    vector<upper=0>[N_transitions] theta;
}

transformed parameters {
    /* transition matrix (log scale) */
    matrix<upper=0>[N_states, N_states] logP; 

    /* Weibull location parameters and shape for residence times */
    array[N_fish] vector<lower=0>[N_transitions] lambda_res;
    
    /* compute residence time parameters for each fish */
    for (i in 1:N_fish) {
        /* regression model for mean residence times */
        vector[N_transitions] mu_res = exp(alpha_res + beta_res * log_fl[i]);
	/* 
	 * coverting the parametrization of the residence
	 * time Weibull distributions from the (mu, kappa)
	 * to the (lambda, kappa) parametrization used by
	 * Stan
	 */
	lambda_res[i] = mu_res ./ tgamma(1 + 1/kappa_res);
    }
    
    /* 
     * populate transition probability matrix of the
     * embedded Markov chain with the theta parameters
     * (only for the allowed transitions)
     */
    { 
        matrix[N_states, N_states] tmp = rep_matrix(negative_infinity(), N_states, N_states);
        for (i in (N_transient+1):N_states) {
            tmp[i,i] = 0;
        }
        for (i in 1:N_transitions) {
            tmp[transitions[1, i], transitions[2, i]] = theta[i];
        }
        /* enforce the stochastic matrix property */
        for (i in 1:N_states) {
            tmp[i, :] = to_row_vector(softmax(to_vector(tmp[i, :])));
        }
	logP = log(tmp);
    }

}


model {

    theta ~ std_normal();

    alpha_res ~ cauchy(1, 2);
    beta_res ~ cauchy(0, 2);
    kappa_res ~ cauchy(0, 2);

    /* probabilities of individual observations */
    for (i in 1:N_obs) {
        row_vector[N_states] lp = logP[curr[i], :] + log_next[i];
        for (j in 1:N_transitions) {
	    real f;
	    if (transitions[1, j] != curr[i]) continue;
	    if (!next[i, transitions[2, j]]) continue;
	    f = weibull_lpdf(tau[i] | kappa_res[j], lambda_res[fish[i]][j]);
	    lp[transitions[2, j]] += f; 
	}
	target += log_sum_exp(lp);
    }

}
