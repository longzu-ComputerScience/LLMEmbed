import unittest

from llmembed_research.results import METHODS, SEEDS, recompute


class ResultsTests(unittest.TestCase):
    def test_reproduces_all_source_and_filtered_metrics(self):
        evidence = recompute()
        self.assertEqual(len(evidence), len(METHODS) * len(SEEDS) * 2)
        for subset in ("source", "filtered"):
            for seed in SEEDS:
                candidate = evidence[METHODS[-1], seed, subset]
                for baseline in METHODS[:2]:
                    for metric, value in candidate.items():
                        self.assertGreater(value, evidence[baseline, seed, subset][metric])


class StudyTests(unittest.TestCase):
    def test_tau_one_matches_weighted_softmax_denominator(self):
        import torch
        from llmembed_research.objective import adjusted_cross_entropy, log_train_prior

        logits = torch.tensor([[1.0, 0.2], [0.7, -0.1]], requires_grad=True)
        labels = torch.tensor([0, 1])
        prior = log_train_prior(torch.tensor([0, 0, 0, 1]), 2)
        expected = torch.nn.functional.cross_entropy(logits + torch.tensor([3.0, 1.0]).log(), labels, reduction="sum")
        loss = adjusted_cross_entropy(logits, labels, prior)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.argmax(dim=1).tolist(), [0, 0])

    def test_uniform_two_level_weights_assign_equal_backbone_mass(self):
        import torch
        from llmembed_research.fusion import hierarchical_gate_weights

        sources = torch.ones(2, 7, 4)
        weights = hierarchical_gate_weights(sources, lambda x: torch.zeros_like(x[..., :1]))
        self.assertEqual(weights.shape, (2, 7, 1))
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(2, 1))
        torch.testing.assert_close(weights[:, :5].sum(dim=1), torch.full((2, 1), 1 / 3))


if __name__ == "__main__":
    unittest.main()
