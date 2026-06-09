# Distributed TD3 Control for Connected MSD Systems

Standardized version of the connected MSD experiments.

- Custom Gymnasium environment for the linear and inverted-sine connected MSD systems.
- Tianshou TD3 for baseline training.
- Minimal validation in `validate_minimal.py`.

Install:

```powershell
pip install -r requirements.txt
```

Validate:

```powershell
python validate_minimal.py
```

Example training command:

```powershell
python train_baseline_td3.py --env_type sine --N 5 --graph complete --m default --d default --k default --dt 0.02 --episode_len 250 --umax 12.0 --q_weight 1.0 --v_weight 0.3 --u_weight 0.03 --terminal_weight 0.0 --q_init_low -2.4 --q_init_high 2.4 --v_init_low -0.4 --v_init_high 0.4 --actor_mode residual --linear_init=-4.8,-1.2,0.0 --action_scale 12.0 --actor_hidden_layers 64,64,64 --actor_activation elu --critic_hidden_layers 128,128,128 --critic_activation elu --actor_lr 5e-6 --critic_lr 1e-4 --batch_size 512 --total_steps 100000 --start_steps 0 --eval_every 500 --eval_episodes 20 --early_stop_final_norm 0.05 --early_stop_patience 3 --best_patience 80 --best_min_delta 0.1 --gamma 0.98 --tau 0.005 --exploration_noise 0.05 --policy_noise 0.05 --policy_delay 20 --save_dir results_sine_inverted_complete
```

Additional experiments, evaluation, and comparison commands are collected in
`command.txt`.
