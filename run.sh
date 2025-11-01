# Ant-2d
nohup python main.py --env_id "MO-Ant-v2" --seed 1   --old_Q_update_freq 1   --regular_bar  0.25   --consider_other   --regular_alpha 0.001  --prefer 0 --buf_num 0 --q_freq 1000 --EA_policy_num 0 --RL_policy_num 0  --iso_sigma 0.005  --line_sigma 0.05 --Use_Critic_Preference  --Use_Policy_Preference     --train_with_fixed_preference    --latent_dim  50   --reward_coef 1.0 --dynamic_coef 1.0 --value_coef 1.0  --encoder_update_freq 1  --Policy_use_latent   --Policy_use_s  --Policy_use_w  --Critic_use_both   --Critic_use_s  --Critic_use_a --use_avg  > ./logs/1.log 2>&1 &
# Half-2d
nohup python main.py --env_id "MO-New-HalfCheetah-v2" --seed 1  --old_Q_update_freq 1   --regular_bar  0.25   --consider_other   --regular_alpha 0.01  --prefer 0 --buf_num 0 --q_freq 1000 --EA_policy_num 0 --RL_policy_num 0  --iso_sigma 0.005  --line_sigma 0.05 --Use_Critic_Preference  --Use_Policy_Preference     --train_with_fixed_preference    --latent_dim  50   --reward_coef 1.0 --dynamic_coef 1.0 --value_coef 1.0  --encoder_update_freq 1  --Policy_use_latent   --Policy_use_s  --Policy_use_w  --Critic_use_both   --Critic_use_s  --Critic_use_a --use_avg  > ./logs/2.log 2>&1 &
# Hopper-2d

nohup python main.py --env_id "MO-Hopper-v2" --seed 1 --old_Q_update_freq 1 --regular_bar 0.25 --consider_other --regular_alpha 1.0 --prefer 0 --buf_num 0 --q_freq 1000 --EA_policy_num 0 --RL_policy_num 0 --iso_sigma 0.005 --line_sigma 0.05 --Use_Critic_Preference --Use_Policy_Preference --train_with_fixed_preference --latent_dim 10 --reward_coef 1.0 --dynamic_coef 1.0 --value_coef 1.0 --encoder_update_freq 1 --Policy_use_latent --Policy_use_s --Policy_use_w --Critic_use_both --Critic_use_s --Critic_use_a  > ./logs/3.log 2>&1 &
# Walker-2d
nohup python main.py --env_id "MO-Walker2d-v2" --seed 1 --old_Q_update_freq 1 --regular_bar 0.25 --consider_other --regular_alpha 1.0 --prefer 0 --buf_num 0 --q_freq 1000 --EA_policy_num 0 --RL_policy_num 0 --iso_sigma 0.005 --line_sigma 0.05 --Use_Critic_Preference --Use_Policy_Preference --train_with_fixed_preference --latent_dim 10 --reward_coef 1.0 --dynamic_coef 1.0 --value_coef 1.0 --encoder_update_freq 1 --Policy_use_latent --Policy_use_s --Policy_use_w --Critic_use_both --Critic_use_s --Critic_use_a  > ./logs/4.log 2>&1 &

# Hopper-3d
nohup python main.py --env_id "MO-Hopper-v3" --seed 1  --old_Q_update_freq 1 --regular_bar 0.25 --consider_other --regular_alpha 1.0 --prefer 0 --buf_num 0 --q_freq 1000 --EA_policy_num 0 --RL_policy_num 0 --iso_sigma 0.005 --line_sigma 0.05 --Use_Critic_Preference --Use_Policy_Preference --train_with_fixed_preference --latent_dim 10 --reward_coef 1.0 --dynamic_coef 1.0 --value_coef 1.0 --encoder_update_freq 1 --Policy_use_latent --Policy_use_s --Policy_use_w --Critic_use_both --Critic_use_s --Critic_use_a  > ./logs/5.log 2>&1 &

# Ant-3d
nohup python main.py --env_id "MO-Ant-v3" --seed 1  --old_Q_update_freq 1 --regular_bar 0.25 --consider_other --regular_alpha 1.0 --prefer 0 --buf_num 0 --q_freq 1000 --EA_policy_num 0 --RL_policy_num 0 --iso_sigma 0.005 --line_sigma 0.05 --Use_Critic_Preference --Use_Policy_Preference --train_with_fixed_preference --latent_dim 50 --reward_coef 1.0 --dynamic_coef 1.0 --value_coef 1.0 --encoder_update_freq 1 --Policy_use_latent --Policy_use_s --Policy_use_w --Critic_use_both --Critic_use_s --Critic_use_a --use_avg     > ./logs/6.log 2>&1 &


# Ant-4d
nohup python  main.py --env_id "MO-Ant-v4" --seed 1 --old_Q_update_freq 1 --regular_bar 0.25 --consider_other --regular_alpha 1.0 --prefer 0 --buf_num 0 --q_freq 1000 --EA_policy_num 0 --RL_policy_num 0 --iso_sigma 0.005 --line_sigma 0.05 --Use_Critic_Preference --Use_Policy_Preference --train_with_fixed_preference --latent_dim 50 --reward_coef 1.0 --dynamic_coef 1.0 --value_coef 1.0 --encoder_update_freq 1 --Policy_use_latent --Policy_use_s --Policy_use_w --Critic_use_both --Critic_use_s --Critic_use_a --use_avg > ./logs/7.log 2>&1 &

# Half-5d

nohup python  main.py --env_id "MO-HalfCheetah-v5" --seed 1 --old_Q_update_freq 1 --regular_bar 0.25 --consider_other --regular_alpha 1.0 --prefer 0 --buf_num 0 --q_freq 1000 --EA_policy_num 0 --RL_policy_num 0 --iso_sigma 0.005 --line_sigma 0.05 --Use_Critic_Preference --Use_Policy_Preference --train_with_fixed_preference --latent_dim 50 --reward_coef 1.0 --dynamic_coef 1.0 --value_coef 1.0 --encoder_update_freq 1 --Policy_use_latent --Policy_use_s --Policy_use_w --Critic_use_both --Critic_use_s --Critic_use_a --use_avg > ./logs/8.log 2>&1 &


# Hopper-5d

nohup python  main.py --env_id "MO-Hopper-v5" --seed 1 --old_Q_update_freq 1 --regular_bar 0.25 --consider_other --regular_alpha 0.01 --prefer 0 --buf_num 0 --q_freq 1000 --EA_policy_num 0 --RL_policy_num 0 --iso_sigma 0.005 --line_sigma 0.05 --Use_Critic_Preference --Use_Policy_Preference --train_with_fixed_preference --latent_dim 50 --reward_coef 1.0 --dynamic_coef 1.0 --value_coef 1.0 --encoder_update_freq 1 --Policy_use_latent --Policy_use_s --Policy_use_w --Critic_use_both --Critic_use_s --Critic_use_a  > ./logs/9.log 2>&1 &

# Ant-5d
nohup python  main.py --env_id "MO-Ant-v5" --seed 1 --old_Q_update_freq 1 --regular_bar 0.25 --consider_other --regular_alpha 1.0 --prefer 0 --buf_num 0 --q_freq 1000 --EA_policy_num 0 --RL_policy_num 0 --iso_sigma 0.005 --line_sigma 0.05 --Use_Critic_Preference --Use_Policy_Preference --train_with_fixed_preference --latent_dim 50 --reward_coef 1.0 --dynamic_coef 1.0 --value_coef 1.0 --encoder_update_freq 1 --Policy_use_latent --Policy_use_s --Policy_use_w --Critic_use_both --Critic_use_s --Critic_use_a --use_avg > ./logs/10.log 2>&1 &

