from collections import defaultdict
from env.portfolio_env import PortfolioEnv
import util


def train(data_df, agent, num_episodes, limit_iterations, num_warmup_iterations, volatility_lookback,
          log_interval_steps, log_comet, comet_log_level, experiment, checkpoints_interval,
          checkpoints_dir, save_checkpoints, is_test=False, results_dir=None):

    data = data_df.to_numpy()

    agent.is_training = not is_test
    phase = 'test' if is_test else 'train'

    num_days = data.shape[0]

    # init custom OpenAI gym env for stocks portfolio
    env = PortfolioEnv(data, volatiltiy_lookback=volatility_lookback)

    output = []

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

            if not is_test and total_iterations_counter < num_warmup_iterations:
                # warmup to fill up the buffer with random actions
                current_action = agent.select_random_action()
            else:
                # regular training. Let agent select action based on observation
                current_action = agent.select_action(current_state)

            if is_test:
                output.append(current_action)

            # execute action on environment, observe new state and reward
            next_state, current_reward, done, _ = env.step(current_action)

            # logging
            results['reward'].append(current_reward)
            results['current_volatility'].append(env.current_volatility)
            results['current_gains'].append(env.current_gains)
            if t % log_interval_steps == 0:
                avg_reward = util.avg_results(results, 'reward', lookback=log_interval_steps)
                avg_vol = util.avg_results(results, 'current_volatility', lookback=log_interval_steps)
                avg_gains = util.avg_results(results, 'current_gains', lookback=log_interval_steps)
                total_gains = env.total_gains

                print('{} episode: {} | step: {} | avg_reward: {:.5f} | avg_vol: {:.2f} | avg_step_gains: {:.2f} | total_gains: {:.2f}'
                    .format(phase, episode, t, avg_reward, avg_vol, avg_gains, total_gains))

                env.render()
                if log_comet and comet_log_level in ['interval']:
                    experiment.log_metric('{}_interval_reward'.format(phase), avg_reward, step=total_iterations_counter)
                    experiment.log_metric('{}_interval_avg_vol'.format(phase), avg_vol, step=total_iterations_counter)
                    experiment.log_metric('{}_interval_avg_gains'.format(phase), avg_gains, step=total_iterations_counter)
                    experiment.log_metric('{}_interval_total_gains'.format(phase), total_gains, step=total_iterations_counter)

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
                    experiment.log_metric('{}_interval_critic_loss'.format(phase), avg_critic_loss, step=total_iterations_counter)
                    experiment.log_metric('{}_interval_actor_loss'.format(phase), avg_actor_loss, step=total_iterations_counter)

            current_state = next_state
            total_iterations_counter += 1

        if limit_iterations is not None and total_iterations_counter >= limit_iterations:
            # option for hard limit on iterations for debugging
            break

        if save_checkpoints and (episode+1) % checkpoints_interval == 0:
            agent.save_model(checkpoints_dir, identifier=episode+1)

        # logging
        avg_reward = util.avg_results(results, 'reward')
        avg_vol = util.avg_results(results, 'current_volatility')
        avg_gains = util.avg_results(results, 'current_gains')
        total_gains = env.total_gains
        avg_critic_loss = util.avg_results(results, 'critic')
        avg_actor_loss = util.avg_results(results, 'actor')

        print('Train episode {} results - reward: {:.2f} | avg_vol: {:.2f} | avg_gains: {:.2f} | total_gains: {:.2f}'
              .format(episode, avg_reward, avg_vol, avg_gains, total_gains))
        if log_comet and comet_log_level in ['episode', 'interval']:

            experiment.log_metric('{}_avg_episode_reward'.format(phase), avg_reward, step=episode)
            experiment.log_metric('{}_avg_episode_critic_loss'.format(phase), avg_critic_loss, step=episode)
            experiment.log_metric('{}_avg_episode_actor_loss'.format(phase), avg_actor_loss, step=episode)
            experiment.log_metric('{}_final_episode_avg_vol'.format(phase), avg_vol, step=episode)
            experiment.log_metric('{}_final_episode_avg_gains'.format(phase), avg_gains, step=episode)
            experiment.log_metric('{}_final_episode_total_gains'.format(phase), total_gains, step=episode)

        env.render()

    if save_checkpoints:
        agent.save_model(checkpoints_dir, identifier='final')

    if is_test:
        util.save_weights(output, columns=data_df.keys(), results_dir=results_dir)


