"""Candidate starter interfaces for the streaming track.

The evaluator instantiates encoder and decoder separately to prevent hidden
side channels. Only emitted symbols are transmitted between them.
"""


class StreamingEncoder:
    def reset(self):
        pass

    def observe(self, observation, previous_action):
        """Return zero or more integer symbols."""
        raise NotImplementedError


class StreamingDecoder:
    def reset(self):
        pass

    def consume(self, symbols):
        """Update decoder state using only transmitted symbols."""
        raise NotImplementedError

    def predict(self, query_action):
        """Predict the next physical consequence for query_action."""
        raise NotImplementedError
