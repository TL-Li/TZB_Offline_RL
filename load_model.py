import numpy as np
import argparse

import offline




if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	# Experiment
	parser.add_argument("--policy", default="offline_policy")               # Policy name
	parser.add_argument("--env", default="TZB")        				# Our environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e4, type=int)   # Max time steps to run environment
	parser.add_argument("--save_model", default= True, action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="default")                 # Model load file name, "" doesn't load, "default" uses file_name
	# offline_policy
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic: default is 256
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--alpha", default=2.5)
	parser.add_argument("--normalize", default=True)
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	


	state_dim = 12
	action_dim = 12
	max_action = 120
	# print(max_action) 119
	# print(action_dim) 12
	# print(state_dim) 12

	kwargs = {
		# base 
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		# offline_policy
		"policy_noise": args.policy_noise * 0,
		"noise_clip": args.noise_clip * max_action,
		"policy_freq": args.policy_freq,
		"alpha": args.alpha
	}
	# state = [118.2, 25.8, 10000, 118.4, 25.8, 10000, 118.6, 25.8, 10000, 118.8, 25.8, 10000]
	state = np.zeros(12)
	# Initialize policy
	policy = offline.offline_policy(**kwargs)
	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")
		ans = policy.select_action(np.array(state))
		ans[2] = ans[5] = ans[8] = ans[11] = 10000 
		print(ans)

	# replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	# replay_buffer.convert_TZB(load_data())
	# if args.normalize:
	# 	mean,std = replay_buffer.normalize_states() 
	# else:
	# 	mean,std = 0,1
	
	# start = time.time()
	# evaluations = []
	# for t in range(int(args.max_timesteps)):
	# 	policy.train(replay_buffer, args.batch_size)
	# 	if t % 1000 == 0:
	# 		# print("It's", t, "epochs")
	# 		end = time.time()
	# 		print("\n Algo {} Exp {} updates {}/{} episodes, total timesteps {}.\n"
	# 	 		.format(args.policy,
	# 					args.env,
	# 					t,
	# 					args.max_timesteps,
	# 					int(end - start)
	# 					))


		
	# 	# Evaluate episode
	# 	if (t + 1) % args.eval_freq == 0:
	# 		print(f"Time steps: {t+1}")
	# 		evaluations.append(eval_policy(policy, args.env, args.seed, mean, std))
	# 		np.save(f"./results/{file_name}", evaluations)
	# 		if args.save_model: policy.save(f"./models/{file_name}")
