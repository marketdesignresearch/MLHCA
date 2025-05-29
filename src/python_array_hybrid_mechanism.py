import os


force_run = True
new_query_option = 'cca'
config = 'hpo1'

# #NOTE: In an ideal world, these should be dynamic 
max_linear_prices_multiplier = 3
start_linear_item_prices = 0
cca_initial_prices_multiplier = 0.7
cca_increment = 0.15


for domain in ['SRVM']:
    for qinit in [20]:
        for init_dq_method in ['cca']:
            for new_query_option in ['gd_linear_prices_on_W_v3']:  # options: gd_linear_prices_on_W_v3, cca, load_prices
            # for new_query_option in ['load_prices']:
                for forbid_individual_bundles in ['True']:
                    for i in range(50):
                    # for i in [10001, 10003, 10018]:
                        path_addition = f'cca_initial_prices_multiplier_{cca_initial_prices_multiplier}_increment_{cca_increment}_'
                        path_str = f'results/{domain}_qinit_{qinit}_initial_demand_query_method_{init_dq_method}_{path_addition}new_query_option_{new_query_option}/ML_config_{config}/{10001 + i}'
                        if (not os.path.exists(path_str)) or force_run:
                            # command = f'sbatch --job-name={domain}B_{qinit} server_script_hybrid_mechanism.sh {domain} {init_dq_method} {qinit} {i} {new_query_option} {forbid_individual_bundles}'
                            command = f'sbatch --job-name={domain}FF{qinit} server_script_hybrid_mechanism.sh {domain} {init_dq_method} {qinit} {10001 + i} {new_query_option} {forbid_individual_bundles}'
                            # print(command)
                            os.system(command)
                        else: 
                            print(f'skipping domain: {domain} qinit: {qinit} seed: {10001 + i}')

