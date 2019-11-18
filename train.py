from env.portfolio_env import PortfolioEnv
from collections import defaultdict
import util


def train(data, agent, init_cash, num_episodes, limit_iterations, num_warmup_iterations,
          log_interval_steps, log_comet, comet_log_level, experiment):
    num_days = data.shape[0]

    # init custom OpenAI gym env for stocks portfolio
    env = PortfolioEnv(data, init_cash)

    # training
    total_iterations_counter = 0 # counter for total iterations. num_episodes * num_days
    for episode in range(num_episodes):
        agent.reset_action_noise_process()  # init random process for new episode
        current_state = env.reset()  # get initial state s(t)

        results = defaultdict(list) # for logging
        for t in range(num_days - 1):
            if limit_iterations is not None and total_iterations_counter >= limit_iterations:
                # option for hard limit on iterations for debugging
                break

            if total_iterations_counter < num_warmup_iterations:
                # warmup to fill up the buffer with random actions
                current_action = agent.select_random_action()
            else:
                # regular training. Let agent select action based on observation
                current_action = agent.select_action(current_state)

            # execute action on environment, observe new state and reward
            next_state, current_reward, done, _ = env.step(current_action)

            # logging
            results['reward'].append(current_reward)
            results['purchase_power'].append(env.current_purchase_power)
            results['profit'].append(env.current_purchase_power - env.init_cash)
            if t % log_interval_steps == 0:
                avg_reward = util.avg_results(results, 'reward', lookback=log_interval_steps)
                avg_ppwr = util.avg_results(results, 'purchase_power', lookback=log_interval_steps)
                avg_profit = util.avg_results(results, 'profit', lookback=log_interval_steps)

                print('Train episode: %d | step: %d | reward: %2f' % (episode, t, avg_reward))
                env.render()
                if log_comet and comet_log_level in ['interval']:
                    experiment.log_metric('train_interval_reward', avg_reward, step=total_iterations_counter)
                    experiment.log_metric('train_interval_ppwr', avg_ppwr, step=total_iterations_counter)
                    experiment.log_metric('train_interval_profit', avg_profit, step=total_iterations_counter)

            # TODO: might need to add episode done states to limit batches not to cross over episodes

            if total_iterations_counter >= num_warmup_iterations:
                # we only want to update the policy after the random state warmup

                # store transition in R (s(t), a(t), r(t), s(t+1))
                agent.append_observation(current_state, current_action, current_reward, next_state)

                # update policy
                critic_loss_val, actor_loss_val = agent.update_policy()

                # logging
                results['critic'].append(critic_loss_val)
                results['actor'].append(actor_loss_val)
                if log_comet and comet_log_level in ['interval']:
                    avg_critic_loss = util.avg_results(results, 'critic', lookback=log_interval_steps)
                    avg_actor_loss = util.avg_results(results, 'actor', lookback=log_interval_steps)
                    experiment.log_metric('train_interval_critic_loss', avg_critic_loss, step=total_iterations_counter)
                    experiment.log_metric('train_interval_actor_loss', avg_actor_loss, step=total_iterations_counter)

            current_state = next_state
            total_iterations_counter += 1

        if limit_iterations is not None and total_iterations_counter >= limit_iterations:
            # option for hard limit on iterations for debugging
            break

        # logging
        print('Train episode: %d final results:' % episode)
        if log_comet and comet_log_level in ['episode', 'interval']:
            avg_reward = util.avg_results(results, 'reward')
            avg_critic_loss = util.avg_results(results, 'critic')
            avg_actor_loss = util.avg_results(results, 'actor')
            experiment.log_metric('train_avg_episode_reward', avg_reward, step=episode)
            experiment.log_metric('train_avg_episode_critic_loss', avg_critic_loss, step=episode)
            experiment.log_metric('train_avg_episode_actor_loss', avg_actor_loss, step=episode)
            experiment.log_metric('train_max_episode_ppwr', env.max_purchase_power, step=episode)
            experiment.log_metric('train_max_episode_profit', env.max_purchase_power - env.init_cash, step=episode)
            experiment.log_metric('train_final_episode_ppwr', env.current_purchase_power, step=episode)
            experiment.log_metric('train_final_episode_profit', env.current_purchase_power - env.init_cash, step=episode)

        env.render()



