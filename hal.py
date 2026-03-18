from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Embeddings pour retrouver le passage le plus proche
from sentence_transformers import SentenceTransformer

# NLI : entailment / neutral / contradiction
from transformers import pipeline


@dataclass
class ClaimResult:
    claim: str
    best_evidence: str
    retrieval_score: float
    label: str
    entailment_score: float
    contradiction_score: float
    neutral_score: float
    supported: bool
    hallucination_risk: float


@dataclass
class EvaluationResult:
    answer: str
    claims: List[ClaimResult]
    faithfulness_score: float
    unsupported_claim_rate: float
    contradiction_rate: float
    avg_hallucination_risk: float


class HallucinationEvaluator:
    """
    Évaluateur simple de hallucinations pour réponses générées.

    Pipeline:
    1. Découper la réponse en claims
    2. Trouver le meilleur passage de contexte pour chaque claim
    3. Appliquer NLI(claim <- evidence)
    4. Calculer des métriques globales
    """

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        nli_model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        support_threshold: float = 0.55,
        contradiction_threshold: float = 0.50,
        min_retrieval_score: float = 0.25,
    ) -> None:
        """
        Args:
            embedding_model_name: modèle d'embeddings pour rapprocher claim/passages
            nli_model_name: modèle NLI
            support_threshold: seuil d'entailment pour dire qu'un claim est supporté
            contradiction_threshold: seuil de contradiction
            min_retrieval_score: score minimal de similarité pour considérer une preuve exploitable
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.nli = pipeline("text-classification", model=nli_model_name, return_all_scores=True)

        self.support_threshold = support_threshold
        self.contradiction_threshold = contradiction_threshold
        self.min_retrieval_score = min_retrieval_score

    @staticmethod
    def split_into_passages(context: str, max_chars: int = 500) -> List[str]:
        """
        Découpe le contexte en passages courts pour la recherche de preuve.
        """
        raw_parts = re.split(r"\n\s*\n|(?<=[.!?])\s+", context)
        raw_parts = [p.strip() for p in raw_parts if p.strip()]

        passages: List[str] = []
        current = ""

        for part in raw_parts:
            if len(current) + len(part) + 1 <= max_chars:
                current = f"{current} {part}".strip()
            else:
                if current:
                    passages.append(current)
                current = part

        if current:
            passages.append(current)

        return passages

    @staticmethod
    def split_answer_into_claims(answer: str) -> List[str]:
        """
        Heuristique simple:
        - découpe en phrases
        - enlève les phrases trop courtes
        """
        sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
        claims = []

        for s in sentences:
            s = s.strip()
            if not s:
                continue

            # On ignore les phrases trop courtes / peu informatives
            if len(s.split()) < 4:
                continue

            claims.append(s)

        return claims

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        return self.embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def retrieve_best_evidence(self, claim: str, passages: List[str]) -> tuple[str, float]:
        """
        Trouve le passage le plus similaire au claim.
        """
        if not passages:
            return "", 0.0

        claim_emb = self._embed_texts([claim])
        passage_embs = self._embed_texts(passages)

        sims = cosine_similarity(claim_emb, passage_embs)[0]
        best_idx = int(np.argmax(sims))
        return passages[best_idx], float(sims[best_idx])

    def run_nli(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """
        Applique le modèle NLI.

        Convention:
        - premise = evidence
        - hypothesis = claim
        """
        # Beaucoup de pipelines NLI attendent "premise [SEP] hypothesis"
        text = f"{premise} [SEP] {hypothesis}"
        outputs = self.nli(text)[0]

        scores = {o["label"].lower(): float(o["score"]) for o in outputs}

        # Harmonisation de labels possibles selon les modèles
        entailment = scores.get("entailment", 0.0)
        contradiction = scores.get("contradiction", 0.0)
        neutral = scores.get("neutral", 0.0)

        return {
            "entailment": entailment,
            "contradiction": contradiction,
            "neutral": neutral,
        }

    def evaluate_claim(self, claim: str, passages: List[str]) -> ClaimResult:
        """
        Évalue un claim individuel.
        """
        evidence, retrieval_score = self.retrieve_best_evidence(claim, passages)

        if retrieval_score < self.min_retrieval_score or not evidence:
            return ClaimResult(
                claim=claim,
                best_evidence=evidence,
                retrieval_score=retrieval_score,
                label="unsupported",
                entailment_score=0.0,
                contradiction_score=0.0,
                neutral_score=1.0,
                supported=False,
                hallucination_risk=1.0,
            )

        nli_scores = self.run_nli(premise=evidence, hypothesis=claim)

        entailment = nli_scores["entailment"]
        contradiction = nli_scores["contradiction"]
        neutral = nli_scores["neutral"]

        if contradiction >= self.contradiction_threshold:
            label = "contradiction"
            supported = False
            hallucination_risk = min(1.0, 0.7 + contradiction * 0.3)
        elif entailment >= self.support_threshold:
            label = "supported"
            supported = True
            hallucination_risk = max(0.0, 1.0 - entailment)
        else:
            label = "unsupported"
            supported = False
            hallucination_risk = max(contradiction, 1.0 - entailment)

        return ClaimResult(
            claim=claim,
            best_evidence=evidence,
            retrieval_score=retrieval_score,
            label=label,
            entailment_score=entailment,
            contradiction_score=contradiction,
            neutral_score=neutral,
            supported=supported,
            hallucination_risk=float(hallucination_risk),
        )

    def evaluate_answer(self, answer: str, context: str) -> EvaluationResult:
        """
        Évalue une réponse complète par rapport à un contexte.
        """
        claims = self.split_answer_into_claims(answer)
        passages = self.split_into_passages(context)

        if not claims:
            return EvaluationResult(
                answer=answer,
                claims=[],
                faithfulness_score=1.0,
                unsupported_claim_rate=0.0,
                contradiction_rate=0.0,
                avg_hallucination_risk=0.0,
            )

        results = [self.evaluate_claim(claim, passages) for claim in claims]

        n = len(results)
        n_supported = sum(1 for r in results if r.supported)
        n_unsupported = sum(1 for r in results if r.label == "unsupported")
        n_contradiction = sum(1 for r in results if r.label == "contradiction")

        faithfulness_score = n_supported / n
        unsupported_claim_rate = n_unsupported / n
        contradiction_rate = n_contradiction / n
        avg_hallucination_risk = float(np.mean([r.hallucination_risk for r in results]))

        return EvaluationResult(
            answer=answer,
            claims=results,
            faithfulness_score=faithfulness_score,
            unsupported_claim_rate=unsupported_claim_rate,
            contradiction_rate=contradiction_rate,
            avg_hallucination_risk=avg_hallucination_risk,
        )


def pretty_print_result(result: EvaluationResult) -> None:
    print("=" * 80)
    print("RÉSUMÉ GLOBAL")
    print("=" * 80)
    print(f"Faithfulness score      : {result.faithfulness_score:.3f}")
    print(f"Unsupported claim rate  : {result.unsupported_claim_rate:.3f}")
    print(f"Contradiction rate      : {result.contradiction_rate:.3f}")
    print(f"Avg hallucination risk  : {result.avg_hallucination_risk:.3f}")
    print()

    print("=" * 80)
    print("DÉTAIL PAR CLAIM")
    print("=" * 80)
    for i, claim_result in enumerate(result.claims, start=1):
        print(f"[{i}] Claim                : {claim_result.claim}")
        print(f"    Label                : {claim_result.label}")
        print(f"    Retrieval score      : {claim_result.retrieval_score:.3f}")
        print(f"    Entailment score     : {claim_result.entailment_score:.3f}")
        print(f"    Contradiction score  : {claim_result.contradiction_score:.3f}")
        print(f"    Neutral score        : {claim_result.neutral_score:.3f}")
        print(f"    Hallucination risk   : {claim_result.hallucination_risk:.3f}")
        print(f"    Best evidence        : {claim_result.best_evidence}")
        print()


if __name__ == "__main__":
    context = """
    Marie Curie est née le 7 novembre 1867 à Varsovie.
    Elle a obtenu le prix Nobel de physique en 1903 avec Pierre Curie et Henri Becquerel.
    Elle a reçu un second prix Nobel, en chimie, en 1911.
    Elle a mené des recherches pionnières sur la radioactivité.
    """

    answer = """
    Marie Curie est née à Varsovie en 1867.
    Elle a reçu le prix Nobel de physique en 1903.
    Elle a obtenu un prix Nobel de chimie en 1911.
    Elle a inventé la relativité générale.
    """

    evaluator = HallucinationEvaluator()
    result = evaluator.evaluate_answer(answer=answer, context=context)
    pretty_print_result(result)
