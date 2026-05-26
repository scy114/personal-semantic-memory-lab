import unittest

from tools.proposals.claim_attribution_router import annotate_claim_attribution


def row(
    proposal_id: str,
    output_kind: str,
    text: str,
    *,
    fact_text: str = "",
    hypothesis_text: str = "",
) -> dict:
    return {
        "proposal_id": proposal_id,
        "proposal_run_id": "run:test",
        "output_kind": output_kind,
        "source_text_quote": text,
        "fact_candidate_text": fact_text,
        "hypothesis_text": hypothesis_text,
        "evidence_refs": ["e:1"],
        "raw_backpointer_refs": [{"locator": "x"}],
        "write_permission": False,
    }


class ClaimAttributionRouterTests(unittest.TestCase):
    def test_clean_intrinsic_fact_is_eligible_for_fact_promotion_review(self):
        annotation = annotate_claim_attribution(
            row(
                "p:intrinsic",
                "portrait_fact_candidate",
                "I prefer quiet focused work.",
                fact_text="The subject prefers quiet focused work.",
            )
        )

        self.assertEqual(annotation["claim_type_hint"], "subject_intrinsic_fact")
        self.assertEqual(annotation["fact_promotion_eligibility"], "eligible")
        self.assertEqual(annotation["promotion_path"], "fact_promotion_review_queue")
        self.assertFalse(annotation["write_permission"])

    def test_first_person_affinity_phrase_is_intrinsic_fact(self):
        annotation = annotate_claim_attribution(
            row(
                "p:affinity",
                "portrait_fact_candidate",
                "Contemporary dance is so expressive and graceful - it really speaks to me.",
                fact_text="Gina likes contemporary dance because it is expressive and graceful.",
            )
        )

        self.assertEqual(annotation["claim_type_hint"], "subject_intrinsic_fact")
        self.assertEqual(annotation["fact_promotion_eligibility"], "eligible")
        self.assertEqual(annotation["promotion_path"], "fact_promotion_review_queue")

    def test_belief_about_node_is_not_fact_promotion_eligible(self):
        annotation = annotate_claim_attribution(
            row(
                "p:belief",
                "portrait_fact_candidate",
                "Xiao Hong, you are really smart.",
                fact_text="Xiao Hong is really smart.",
            )
        )

        self.assertEqual(annotation["claim_type_hint"], "subject_belief_about_node")
        self.assertEqual(annotation["fact_promotion_eligibility"], "ineligible")
        self.assertEqual(annotation["promotion_path"], "hypothesis_pool_candidates")

    def test_relation_to_node_is_preserved_as_relation_candidate(self):
        annotation = annotate_claim_attribution(
            row(
                "p:relation",
                "portrait_hypothesis_candidate",
                "I should learn from you.",
                hypothesis_text="The subject wants to learn from the addressed person.",
            )
        )

        self.assertEqual(annotation["claim_type_hint"], "subject_relation_to_node")
        self.assertEqual(annotation["fact_promotion_eligibility"], "ineligible")
        self.assertEqual(annotation["promotion_path"], "future_relation_candidate")

    def test_interaction_hypothesis_goes_to_hypothesis_pool(self):
        annotation = annotate_claim_attribution(
            row(
                "p:interaction",
                "portrait_hypothesis_candidate",
                "Gina encourages Jon in this exchange.",
                hypothesis_text="Gina encourages Jon in this exchange.",
            )
        )

        self.assertEqual(annotation["claim_type_hint"], "interaction_hypothesis")
        self.assertEqual(annotation["promotion_path"], "hypothesis_pool_candidates")

    def test_ambiguous_attribution_goes_to_uncertain_pool(self):
        annotation = annotate_claim_attribution(
            row(
                "p:ambiguous",
                "portrait_fact_candidate",
                "That is great. We should do it.",
                fact_text="The subject should do it.",
            )
        )

        self.assertEqual(annotation["claim_type_hint"], "unknown")
        self.assertEqual(annotation["fact_promotion_eligibility"], "uncertain")
        self.assertEqual(annotation["promotion_path"], "uncertain_pool")

    def test_no_useful_modeling_value_has_no_promotion(self):
        annotation = annotate_claim_attribution(row("p:none", "reject", "Okay."))

        self.assertEqual(annotation["claim_type_hint"], "no_useful_modeling_value")
        self.assertEqual(annotation["promotion_path"], "no_promotion")

    def test_other_node_objective_mispromotion_is_blocked(self):
        annotation = annotate_claim_attribution(
            row(
                "p:mispromote",
                "portrait_fact_candidate",
                "Xiao Hong is smart.",
                fact_text="Xiao Hong is smart.",
            )
        )

        self.assertEqual(annotation["claim_type_hint"], "subject_belief_about_node")
        self.assertEqual(annotation["fact_promotion_eligibility"], "ineligible")

    def test_tool_project_relation_is_not_intrinsic_fact_promotion(self):
        annotation = annotate_claim_attribution(
            row(
                "p:tool",
                "portrait_fact_candidate",
                "I use tool X for this project.",
                fact_text="The subject uses tool X for this project.",
            )
        )

        self.assertEqual(annotation["claim_type_hint"], "subject_relation_to_node")
        self.assertEqual(annotation["claim_target_node_type"], "tool")
        self.assertEqual(annotation["fact_promotion_eligibility"], "ineligible")


if __name__ == "__main__":
    unittest.main()
