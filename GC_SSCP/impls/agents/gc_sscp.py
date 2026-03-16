from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, GCValue, Identity, LengthNormalize, ShortcutActorVectorField

class GCSSCP(flax.struct.PyTreeNode):
    """Hierarchical implicit Q-learning (HIQL) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        """Compute the IVL value loss.

        This value loss is similar to the original IQL value loss, but involves additional tricks to stabilize training.
        For example, when computing the expectile loss, we separate the advantage part (which is used to compute the
        weight) and the difference part (which is used to compute the loss), where we use the target value function to
        compute the former and the current value function to compute the latter. This is similar to how double DQN
        mitigates overestimation bias.
        """
        (next_v1_t, next_v2_t) = self.network.select('target_value')(batch['next_observations'], batch['value_goals'])
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t

        (v1_t, v2_t) = self.network.select('target_value')(batch['observations'], batch['value_goals'])
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1_t
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2_t
        (v1, v2) = self.network.select('value')(batch['observations'], batch['value_goals'], params=grad_params)
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def low_actor_loss(self, batch, grad_params):
        """Compute the low-level actor loss."""
        key, subkey1, subkey2 = jax.random.split(self.rng, 3)
        if self.config['policy_extr'] == 'BC':
            adv = jnp.zeros_like(batch['rewards'])
        elif self.config['policy_extr'] == 'AWR':
            v1, v2 = self.network.select('value')(batch['observations'], batch['low_actor_goals'])
            nv1, nv2 = self.network.select('value')(batch['next_observations'], batch['low_actor_goals'])
            v = (v1 + v2) / 2
            nv = (nv1 + nv2) / 2
            adv = nv - v
        else:
            raise ValueError(f'Invalid policy extraction method: {self.config["policy_extr"]}')
        
        exp_a = jnp.exp(adv * self.config['low_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        # Compute the goal representations of the subgoals.
        goal_reps = self.network.select('goal_rep')(
            jnp.concatenate([batch['observations'], batch['low_actor_goals']], axis=-1),
            params=grad_params,
        )
        if not self.config['low_actor_rep_grad']:
            # Stop gradients through the goal representations.
            goal_reps = jax.lax.stop_gradient(goal_reps)
        
        x0_a = jax.random.normal(subkey1, (batch['actions'].shape[0], batch['actions'].shape[1]))
        x0_g = jax.random.normal(subkey2, (batch['actions'].shape[0], self.config['rep_dim']))

        time = jnp.ones((batch['actions'].shape[0], 1))
        step = jnp.ones((batch['actions'].shape[0], 1))

        pred_a, pred_g = self.network.select('high_actor')(
            batch['observations'], goal_reps, x0_a, x0_g, time, step, is_encoded=True, params=grad_params
        )
        x1_a = x0_a + pred_a
        x1_g = x0_g + pred_g

        target_a = batch['actions']
        target_g = self.network.select('goal_rep')(
            jnp.concatenate([batch['observations'], batch['next_observations']], axis=-1)
        )

        actor_loss = (exp_a * (jnp.mean((x1_a - target_a) ** 2, axis=-1) + jnp.mean((x1_g - target_g) ** 2, axis=-1))).mean()


        return actor_loss, {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_loss_action': jnp.mean((pred_a - target_a) ** 2),
            'bc_loss_goal': jnp.mean((pred_g - target_g) ** 2),
        }

    def high_actor_loss(self, batch, grad_params):
        """Compute the high-level actor loss."""
        key, subkey1, subkey2 = jax.random.split(self.rng, 3)
        if self.config['policy_extr'] == 'BC':
            adv = jnp.zeros_like(batch['rewards'])
        elif self.config['policy_extr'] == 'AWR':
            v1, v2 = self.network.select('value')(batch['observations'], batch['high_actor_goals'])
            nv1, nv2 = self.network.select('value')(batch['high_actor_targets'], batch['high_actor_goals'])
            v = (v1 + v2) / 2
            nv = (nv1 + nv2) / 2
            adv = nv - v
        else:
            raise ValueError(f'Invalid policy extraction method: {self.config["policy_extr"]}')

        exp_a = jnp.exp(adv * self.config['high_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        x0_a = jax.random.normal(subkey1, (batch['actions'].shape[0], batch['actions'].shape[1]))
        x0_g = jax.random.normal(subkey2, (batch['actions'].shape[0], self.config['rep_dim']))

        time = jnp.zeros((batch['actions'].shape[0], 1))
        step = jnp.ones((batch['actions'].shape[0], 1))

        goal_reps = self.network.select('goal_rep')(
            jnp.concatenate([batch['observations'], batch['high_actor_goals']], axis=-1),
            params=grad_params,
        )
        if not self.config['high_actor_rep_grad']:
            # Stop gradients through the goal representations.
            goal_reps = jax.lax.stop_gradient(goal_reps)

        pred_a, pred_g = self.network.select('high_actor')(
            batch['observations'], goal_reps, x0_a, x0_g, time, step, is_encoded=True, params=grad_params
        )

        target_a = batch['actions']
        target_g = self.network.select('goal_rep')(
            jnp.concatenate([batch['observations'], batch['high_actor_targets']], axis=-1)
        )

        x1_a = x0_a + pred_a
        x1_g = x0_g + pred_g

        actor_loss = (exp_a * (jnp.mean((x1_a - target_a) ** 2, axis=-1) + jnp.mean((x1_g - target_g) ** 2, axis=-1))).mean()

        return actor_loss, {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_loss_action': jnp.mean((pred_a - target_a) ** 2),
            'bc_loss_goal': jnp.mean((pred_g - target_g) ** 2),
        }
    
    def self_consistency_loss(self, batch, grad_params):
        """Compute the self-consistency loss."""
        
        key, subkey1, subkey2, subkey3, subkey4 = jax.random.split(self.rng, 5)

        # Find the target through the long route
        # Step: 1
        x0_a = jax.random.normal(subkey1, (batch['actions'].shape[0], batch['actions'].shape[1]))
        x0_g = jax.random.normal(subkey2, (batch['actions'].shape[0], self.config['rep_dim']))
        time = jnp.zeros((batch['actions'].shape[0], 1))
        step = jnp.ones((batch['actions'].shape[0], 1))
        pred_a, pred_g = self.network.select('target_high_actor')(
            batch['observations'], batch['high_actor_goals'], x0_a, x0_g, time, step, #is_encoded=True
        )
        pred_a = pred_a + x0_a
        pred_g = pred_g + x0_g
        pred_g = pred_g / jnp.linalg.norm(pred_g, axis=-1, keepdims=True) * jnp.sqrt(pred_g.shape[-1])

        # Step: 2
        x1_a = jax.random.normal(subkey3, (batch['actions'].shape[0], batch['actions'].shape[1]))
        x1_g = jax.random.normal(subkey4, (batch['actions'].shape[0], self.config['rep_dim']))
        time = time+1
        pred_a, pred_g = self.network.select('target_high_actor')(
            batch['observations'], pred_g, x1_a, x1_g, time, step, is_encoded=True
        )
        pred_a = pred_a + x1_a
        pred_g = pred_g + x1_g
        pred_a = jnp.clip(pred_a, -1, 1)
        pred_g = pred_g / jnp.linalg.norm(pred_g, axis=-1, keepdims=True) * jnp.sqrt(pred_g.shape[-1])

        # Stop gradients through the goal representations.
        pred_g = jax.lax.stop_gradient(pred_g)
        pred_a = jax.lax.stop_gradient(pred_a)

        # Find the prediction through the short route
        shortcut_time = time-1
        shortcut_step = step+1
        s_pred_a, s_pred_g = self.network.select('high_actor')(
            batch['observations'], batch['high_actor_goals'], x0_a, x0_g, shortcut_time, shortcut_step, params=grad_params
        )
        s_pred_a = s_pred_a + x0_a
        s_pred_g = s_pred_g + x0_g
        s_pred_a = jnp.clip(s_pred_a, -1, 1)
        s_pred_g = s_pred_g / jnp.linalg.norm(s_pred_g, axis=-1, keepdims=True) * jnp.sqrt(s_pred_g.shape[-1])

        # Compute the self-consistency loss
        pred_a_loss = jnp.mean((pred_a - s_pred_a) ** 2)
        pred_g_loss = jnp.mean((pred_g - s_pred_g) ** 2)
        total_loss = pred_a_loss + pred_g_loss

        # for logging purposes only
        goal_rep_next_obs = self.network.select('goal_rep')(
            jnp.concatenate([batch['observations'], batch['next_observations']], axis=-1)
        )
        mse_next_obs_rep_long = jnp.mean((goal_rep_next_obs - pred_g) ** 2)
        mse_next_obs_rep_short = jnp.mean((goal_rep_next_obs - s_pred_g) ** 2)
        mse_actions_long = jnp.mean((batch['actions'] - pred_a) ** 2)
        mse_actions_short = jnp.mean((batch['actions'] - s_pred_a) ** 2)


        return total_loss, {
            'pred_a_loss': pred_a_loss,
            'pred_g_loss': pred_g_loss,
            'total_loss': total_loss,
            'mse_next_obs_rep_long': mse_next_obs_rep_long,
            'mse_next_obs_rep_short': mse_next_obs_rep_short,
            'mse_actions_long': mse_actions_long,
            'mse_actions_short': mse_actions_short,
        }





    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        low_actor_loss, low_actor_info = self.low_actor_loss(batch, grad_params)
        for k, v in low_actor_info.items():
            info[f'low_actor/{k}'] = v

        high_actor_loss, high_actor_info = self.high_actor_loss(batch, grad_params)
        for k, v in high_actor_info.items():
            info[f'high_actor/{k}'] = v

        self_consistency_loss, self_consistency_info = self.self_consistency_loss(batch, grad_params)
        for k, v in self_consistency_info.items():
            info[f'self_consistency/{k}'] = v

        loss = value_loss + low_actor_loss + high_actor_loss + self_consistency_loss * self.config['consistency_weight']
        return loss, info

    def target_update(self, network, module_name, tau=None):
        """Update the target network."""
        tau = tau if tau is not None else self.config['tau']
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * tau + tp * (1 - tau),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'value')
        self.target_update(new_network, 'high_actor', tau=self.config['actor_tau'])

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """
            Sample actions from the actor.
        """
        high_seed, low_seed = jax.random.split(seed)

        observations = observations.reshape(-1, observations.shape[-1])
        goals = goals.reshape(-1, goals.shape[-1])
        x0_a = jax.random.normal(high_seed, (observations.shape[0], self.config['action_dim']))
        x0_g = jax.random.normal(high_seed, (observations.shape[0], self.config['rep_dim']))
        time = jnp.zeros((observations.shape[0], 1)) + self.config['inference_time']
        step = jnp.zeros((observations.shape[0], 1)) + self.config['inference_step']


        pred0_a, pred0_g = self.network.select('target_high_actor')(
            observations, goals, x0_a, x0_g, time, step
        )
        pred0_a = pred0_a + x0_a
        pred0_g = pred0_g + x0_g
        pred0_g = pred0_g / jnp.linalg.norm(pred0_g, axis=-1, keepdims=True) * jnp.sqrt(pred0_g.shape[-1])

        actions = jnp.clip(pred0_a, -1, 1)
        return actions[0]
    
    @jax.jit
    def sample_actions_hierarchical(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """
            Sample actions from the actor using the longer route.
        """
        high_seed, low_seed = jax.random.split(seed)

        observations = observations.reshape(-1, observations.shape[-1])
        goals = goals.reshape(-1, goals.shape[-1])
        x0_a = jax.random.normal(high_seed, (observations.shape[0], self.config['action_dim']))
        x0_g = jax.random.normal(high_seed, (observations.shape[0], self.config['rep_dim']))
        time = jnp.zeros((observations.shape[0], 1))
        step = jnp.zeros((observations.shape[0], 1)) + 1


        # Step 1
        pred0_a, pred0_g = self.network.select('high_actor')(
            observations, goals, x0_a, x0_g, time, step
        )
        pred0_a = pred0_a + x0_a
        pred0_g = pred0_g + x0_g
        pred0_g = pred0_g / jnp.linalg.norm(pred0_g, axis=-1, keepdims=True) * jnp.sqrt(pred0_g.shape[-1])

        # Step 2
        x1_a = jax.random.normal(high_seed, (observations.shape[0], self.config['action_dim']))
        x1_g = jax.random.normal(high_seed, (observations.shape[0], self.config['rep_dim']))
        time = time + 1
        pred1_a, pred1_g = self.network.select('high_actor')(
            observations, pred0_g, x1_a, x1_g, time, step, is_encoded=True
        )
        pred1_a = pred1_a + x1_a
        pred1_g = pred1_g + x1_g
        
        actions = jnp.clip(pred1_a, -1, 1)
        return actions[0]
        


    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Define (state-dependent) subgoal representation phi([s; g]) that outputs a length-normalized vector.
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            goal_rep_seq = [encoder_module()]
        else:
            goal_rep_seq = []
        goal_rep_seq.append(
            MLP(
                hidden_dims=(*config['value_hidden_dims'], config['rep_dim']),
                activate_final=False,
                layer_norm=config['layer_norm'],
            )
        )
        goal_rep_seq.append(LengthNormalize())
        goal_rep_def = nn.Sequential(goal_rep_seq)

        # Define the encoders that handle the inputs to the value and actor networks.
        # The subgoal representation phi([s; g]) is trained by the parameterized value function V(s, phi([s; g])).
        # The high-level actor predicts the subgoal representation phi([s; w]) for subgoal w given s and g.
        # The low-level actor predicts actions given the current state s and the subgoal representation phi([s; w]).
        if config['encoder'] is not None:
            # Pixel-based environments require visual encoders for state inputs, in addition to the pre-defined shared
            # encoder for subgoal representations.

            # Value: V(encoder^V(s), phi([s; g]))
            value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            target_value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            # Low-level actor: pi^l(. | encoder^l(s), phi([s; w]))
            low_actor_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            # High-level actor: pi^h(. | encoder^h([s; g]))
            high_actor_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            target_high_actor_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
        else:
            # State-based environments only use the pre-defined shared encoder for subgoal representations.

            # Value: V(s, phi([s; g]))
            value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            target_value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            # Low-level actor: pi^l(. | s, phi([s; w]))
            low_actor_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            # High-level actor: pi^h(. | s, g) (i.e., no encoder)
            high_actor_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            target_high_actor_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)

        # Define value and actor networks.
        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=value_encoder_def,
        )
        target_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=target_value_encoder_def,
        )

        high_actor_def = ShortcutActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            time_dim=config['level_dim'],
            goal_rep_dim=config['rep_dim'],
            layer_norm=config['layer_norm'],
            encoder=high_actor_encoder_def,
        )
        target_high_actor_def = ShortcutActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            time_dim=config['level_dim'],
            goal_rep_dim=config['rep_dim'],
            layer_norm=config['layer_norm'],
            encoder=target_high_actor_encoder_def,
        )

        ex_goalreps = jnp.zeros((1, config['rep_dim']))
        ex_times = jnp.zeros((1, 1))
        config['action_dim'] = action_dim


        network_info = dict(
            goal_rep=(goal_rep_def, (jnp.concatenate([ex_observations, ex_goals], axis=-1))),
            value=(value_def, (ex_observations, ex_goals)),
            target_value=(target_value_def, (ex_observations, ex_goals)),
            # inv_kin=(invkin_def, (ex_observations, ex_goals)),
            high_actor=(high_actor_def, (ex_observations, ex_goals, ex_actions, ex_goalreps, ex_times, ex_times)),
            target_high_actor=(target_high_actor_def, (ex_observations, ex_goals, ex_actions, ex_goalreps, ex_times, ex_times)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_value'] = params['modules_value']
        params['modules_target_high_actor'] = params['modules_high_actor']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='gc_sscp',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.7,  # IQL expectile.
            low_alpha=5.0,  # Low-level AWR temperature.
            high_alpha=5.0,  # High-level AWR temperature.
            subgoal_steps=25,  # Subgoal steps.
            rep_dim=10,  # Goal representation dimension.
            level_dim=128,  # Level dimension.
            low_actor_rep_grad=False,  # Whether low-actor gradients flow to goal representation.
            high_actor_rep_grad=True,  # Whether high-actor gradients flow to goal representation.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name.
            #
            actor_tau=0.001,  # Actor target network update rate.
            inference_time=0,  # Inference time.
            inference_step=2,  # Step.
            consistency_weight=1.0,  # Self-consistency loss weight.
            policy_extr='AWR', # Policy extraction method; BC or AWR.
            # Dataset hyperparameters.
            dataset_class='HGCDataset',  # Dataset class name.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
