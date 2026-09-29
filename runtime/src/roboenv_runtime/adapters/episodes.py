"""The owner's episode policy for adapters whose world advances one control interval at a time.

A task may end its episode at the first control interval whose native verdict is complete
(``stop_on_success``), and a finite action-frame budget (the task's ``max_steps`` or the
simulator's own horizon, in a stepped world) ends it once exhausted. The worker enforces both
after every interval the world advances (``SimWorker._check_terminal``), so the episode ends with
a sealed verdict rather than a failed world. This mixin only takes the task's policy and reports
whether the episode is bounded; the native simulators never see the policy.
"""


class EpisodePolicy:
    stop_on_success = False

    def _take_policy(self, stop_on_success):
        if type(stop_on_success) is not bool:
            raise ValueError("stop_on_success must be a boolean")
        self.stop_on_success = stop_on_success

    @property
    def bounded_episodes(self):
        return self.stop_on_success or self.budget().max is not None
