import asyncio
import sys
import os
import json
import sqlite3
from datetime import datetime

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..')
)

from config import config
from core.chat_loop import ChatLoop
from memory.db import MemoryDB
from eval.test_scenarios import TEST_SCENARIOS


class EvaluationHarness:
    def __init__(self):
        self.chat_loop = ChatLoop(config)
        self.memory_db = MemoryDB(config.db_path)

        self.results = {
            "timestamp": datetime.now().isoformat(),
            "total_scenarios": 0,
            "passed": 0,
            "failed": 0,
            "total_turns": 0,
            "passed_turns": 0,
            "failed_turns": 0,
            "scenarios": []
        }

    async def run_all_scenarios(self):
        print("\n" + "=" * 100)
        print("AI COMPANION MEMORY EVALUATION HARNESS")
        print("=" * 100)

        print(f"\nDatabase: {config.db_path}")
        print(f"Response model: {config.model_response}")
        print(f"Logic model: {config.model_logic}")
        print(f"Embedding model: {config.embedding_model}")

        for scenario in TEST_SCENARIOS:
            await self.run_scenario(scenario)

        self.print_results()

    async def run_scenario(self, scenario):
        scenario_name = scenario["name"]

        print("\n" + "=" * 100)
        print(f"SCENARIO: {scenario_name}")
        print("=" * 100)

        scenario_result = {
            "name": scenario_name,
            "passed": True,
            "turns": [],
            "failures": []
        }

        await self.reset_scenario()

        for turn_number, turn in enumerate(
            scenario["turns"],
            1
        ):
            turn_result = await self.run_turn(
                turn_number,
                turn
            )

            scenario_result["turns"].append(
                turn_result
            )

            self.results["total_turns"] += 1

            if turn_result["passed"]:
                self.results["passed_turns"] += 1
            else:
                self.results["failed_turns"] += 1
                scenario_result["passed"] = False
                scenario_result["failures"].append(
                    turn_result
                )

        self.results["total_scenarios"] += 1

        if scenario_result["passed"]:
            self.results["passed"] += 1
        else:
            self.results["failed"] += 1

        status = "[PASS]" if scenario_result["passed"] else "[FAIL]"

        print("\n" + "-" * 100)
        print(f"{status} SCENARIO: {scenario_name}")
        print("-" * 100)

    async def run_turn(
        self,
        turn_number,
        turn
    ):
        check_type = turn.get(
            "check_type",
            "chat"
        )

        user_message = turn["user"]

        print("\n" + "-" * 80)
        print(f"TURN {turn_number}")
        print(f"TYPE: {check_type}")
        print("-" * 80)

        print(f"USER: {user_message}")

        result = {
            "turn": turn_number,
            "type": check_type,
            "user_message": user_message,
            "passed": True
        }

        try:
            response = await self.chat_loop.get_response(
                user_message
            )

            print(f"ALEX: {response}")

            result["response"] = response

        except Exception as e:
            result["passed"] = False
            result["error"] = f"ChatLoop error: {str(e)}"

            print(
                f"[FAIL] ChatLoop crashed: {e}"
            )

            return result

        if check_type == "store":
            result = await self._check_store(
                turn,
                result
            )

        elif check_type == "recall":
            result = await self._check_recall(
                response,
                turn,
                result
            )

        elif check_type == "update":
            result = await self._check_update(
                turn,
                result
            )

        elif check_type == "personality":
            result = await self._check_personality(
                response,
                turn,
                result
            )

        elif check_type == "entity":
            result = await self._check_entity(
                response,
                turn,
                result
            )

        elif check_type == "contamination":
            result = await self._check_contamination(
                response,
                turn,
                result
            )

        elif check_type == "chat":
            print("[PASS] Chat turn executed successfully")

        else:
            result["passed"] = False
            result["error"] = (
                f"Unknown check type: {check_type}"
            )

        await self._print_database_state()

        status = "[PASS]" if result["passed"] else "[FAIL]"

        print(
            f"\n{status} TURN {turn_number}"
        )

        if not result["passed"]:
            print(
                f"ERROR: {result.get('error', 'Unknown error')}"
            )

        return result

    async def _check_store(
        self,
        turn,
        result
    ):
        expected_facts = turn.get(
            "expected_facts",
            []
        )

        all_facts = await self.memory_db.get_all_facts()

        stored_texts = [
            str(
                fact.get(
                    "content",
                    ""
                )
            ).lower()
            for fact in all_facts
        ]

        missing = []

        for expected in expected_facts:
            expected_lower = expected.lower()

            if not any(
                expected_lower in stored
                for stored in stored_texts
            ):
                missing.append(expected)

        result["database_facts"] = all_facts

        if missing:
            result["passed"] = False
            result["error"] = (
                f"Missing facts: {missing}"
            )

            print(
                f"[FAIL] Missing facts: {missing}"
            )

        else:
            print(
                "[PASS] All expected facts stored"
            )

        return result

    async def _check_recall(
        self,
        response,
        turn,
        result
    ):
        expected_recall = turn.get(
            "expected_recall",
            []
        )

        not_recall = turn.get(
            "not_recall",
            []
        )

        response_lower = response.lower()

        missing = []

        for expected in expected_recall:
            if expected.lower() not in response_lower:
                missing.append(expected)

        wrongly_recalled = []

        for forbidden in not_recall:
            if forbidden.lower() in response_lower:
                wrongly_recalled.append(forbidden)

        result["expected_recall"] = expected_recall
        result["not_recall"] = not_recall

        if missing:
            result["missing_recall"] = missing

        if wrongly_recalled:
            result["wrongly_recalled"] = wrongly_recalled

        if missing or wrongly_recalled:
            result["passed"] = False

            errors = []

            if missing:
                errors.append(
                    f"Missing recall: {missing}"
                )

            if wrongly_recalled:
                errors.append(
                    f"Wrong recall: {wrongly_recalled}"
                )

            result["error"] = ", ".join(errors)

            print(
                f"[FAIL] {result['error']}"
            )

        else:
            print(
                "[PASS] Expected information recalled correctly"
            )

        return result

    async def _check_entity(
        self,
        response,
        turn,
        result
    ):
        expected_recall = turn.get(
            "expected_recall",
            []
        )

        not_recall = turn.get(
            "not_recall",
            []
        )

        expected_entity = turn.get(
            "expected_entity"
        )

        response_lower = response.lower()

        missing = []

        for expected in expected_recall:
            if expected.lower() not in response_lower:
                missing.append(expected)

        wrongly_recalled = []

        for forbidden in not_recall:
            if forbidden.lower() in response_lower:
                wrongly_recalled.append(forbidden)

        entity_correct = True

        if expected_entity:
            entity_correct = (
                expected_entity.lower()
                in response_lower
            )

        if missing or wrongly_recalled or not entity_correct:
            result["passed"] = False

            errors = []

            if missing:
                errors.append(
                    f"Missing: {missing}"
                )

            if wrongly_recalled:
                errors.append(
                    f"Wrong entity information: {wrongly_recalled}"
                )

            if not entity_correct:
                errors.append(
                    f"Expected entity: {expected_entity}"
                )

            result["error"] = ", ".join(errors)

            print(
                f"[FAIL] {result['error']}"
            )

        else:
            print(
                "[PASS] Entity and facts correctly recalled"
            )

        return result

    async def _check_contamination(
        self,
        response,
        turn,
        result
    ):
        should_contain = turn.get(
            "expected_recall",
            []
        )

        should_not_contain = turn.get(
            "not_recall",
            []
        )

        response_lower = response.lower()

        missing = [
            value
            for value in should_contain
            if value.lower() not in response_lower
        ]

        contamination = [
            value
            for value in should_not_contain
            if value.lower() in response_lower
        ]

        if missing or contamination:
            result["passed"] = False

            errors = []

            if missing:
                errors.append(
                    f"Missing correct information: {missing}"
                )

            if contamination:
                errors.append(
                    f"Entity contamination: {contamination}"
                )

            result["error"] = ", ".join(errors)

            print(
                f"[FAIL] {result['error']}"
            )

        else:
            print(
                "[PASS] No cross-entity contamination detected"
            )

        return result

    async def _check_update(
        self,
        turn,
        result
    ):
        all_facts = await self.memory_db.get_all_facts()

        expected_new = turn.get(
            "expected_new_fact"
        )

        expected_old = turn.get(
            "expected_old_fact"
        )

        category = turn.get(
            "category"
        )

        entity_name = turn.get(
            "entity_name"
        )

        print(
            f"\n[Update verification]"
        )

        print(
            f"Expected new fact: {expected_new}"
        )

        print(
            f"Expected old fact: {expected_old}"
        )

        if category:
            print(
                f"Expected category: {category}"
            )

        if entity_name:
            print(
                f"Expected entity: {entity_name}"
            )

        if expected_new:
            new_exists = any(
                expected_new.lower()
                in str(
                    fact.get(
                        "content",
                        ""
                    )
                ).lower()
                for fact in all_facts
            )
        else:
            new_exists = True

        if expected_old:
            old_exists = any(
                expected_old.lower()
                in str(
                    fact.get(
                        "content",
                        ""
                    )
                ).lower()
                for fact in all_facts
            )
        else:
            old_exists = False

        if entity_name:
            entity_facts = [
                fact
                for fact in all_facts
                if str(
                    fact.get(
                        "entity_name",
                        ""
                    )
                ).lower()
                ==
                entity_name.lower()
            ]
        else:
            entity_facts = all_facts

        if category:
            category_facts = [
                fact
                for fact in entity_facts
                if str(
                    fact.get(
                        "category",
                        ""
                    )
                ).lower()
                ==
                category.lower()
            ]
        else:
            category_facts = entity_facts

        print(
            f"[Relevant facts count]: "
            f"{len(category_facts)}"
        )

        for fact in category_facts:
            print(
                f"  ID={fact.get('id')} "
                f"ENTITY={fact.get('entity_name')} "
                f"CATEGORY={fact.get('category')} "
                f"FACT={fact.get('content')}"
            )

        duplicated = False

        if expected_new:
            matching_new = [
                fact
                for fact in category_facts
                if expected_new.lower()
                in str(
                    fact.get(
                        "content",
                        ""
                    )
                ).lower()
            ]

            duplicated = len(matching_new) > 1

        if not new_exists:
            result["passed"] = False
            result["error"] = (
                f"New fact was not stored: "
                f"{expected_new}"
            )

        elif old_exists:
            result["passed"] = False
            result["error"] = (
                f"Old fact still exists: "
                f"{expected_old}"
            )

        elif duplicated:
            result["passed"] = False
            result["error"] = (
                "New fact was duplicated"
            )

        else:
            print(
                "[PASS] Fact updated correctly"
            )

        return result

    async def _check_personality(
        self,
        response,
        turn,
        result
    ):
        expected_traits = turn.get(
            "expected_personality",
            []
        )

        not_traits = turn.get(
            "not_personality",
            []
        )

        response_lower = response.lower()

        missing = [
            trait
            for trait in expected_traits
            if trait.lower() not in response_lower
        ]

        contradicted = [
            trait
            for trait in not_traits
            if trait.lower() in response_lower
        ]

        if missing or contradicted:
            result["passed"] = False

            errors = []

            if missing:
                errors.append(
                    f"Missing traits: {missing}"
                )

            if contradicted:
                errors.append(
                    f"Contradicted traits: {contradicted}"
                )

            result["error"] = ", ".join(errors)

            print(
                f"[FAIL] {result['error']}"
            )

        else:
            print(
                "[PASS] Personality response correct"
            )

        return result

    async def _print_database_state(self):
        try:
            all_facts = await self.memory_db.get_all_facts()

            print(
                f"\n[DATABASE STATE] "
                f"{len(all_facts)} facts"
            )

            for fact in all_facts:
                entity = fact.get(
                    "entity_name",
                    "unknown"
                )

                relationship = fact.get(
                    "relationship",
                    ""
                )

                category = fact.get(
                    "category",
                    ""
                )

                content = fact.get(
                    "content",
                    ""
                )

                print(
                    f"  ID={fact.get('id')} "
                    f"| ENTITY={entity} "
                    f"| REL={relationship} "
                    f"| CATEGORY={category} "
                    f"| {content}"
                )

        except Exception as e:
            print(
                f"[Database inspection error]: {e}"
            )

    async def reset_scenario(self):
        print(
            "\n[Resetting scenario memory...]"
        )

        try:
            await self._clear_database()

        except Exception as e:
            print(
                f"[Database reset error]: {e}"
            )
            raise

        self.memory_db = MemoryDB(
            config.db_path
        )

        self.chat_loop = ChatLoop(
            config
        )

        await self.chat_loop.initialize()

    async def _clear_database(self):
        db_path = config.db_path

        if not os.path.exists(db_path):
            return

        connection = sqlite3.connect(
            db_path
        )

        try:
            cursor = connection.cursor()

            tables = [
                "memory_access_log",
                "entity_relationships",
                "facts",
                "entities",
                "personality_traits",
                "conversations"
            ]

            for table in tables:
                try:
                    cursor.execute(
                        f"DELETE FROM {table}"
                    )
                except sqlite3.OperationalError:
                    pass

            try:
                cursor.execute(
                    "DELETE FROM sqlite_sequence"
                )
            except sqlite3.OperationalError:
                pass

            connection.commit()

        finally:
            connection.close()

    def print_results(self):
        print("\n\n" + "=" * 100)
        print("FINAL EVALUATION RESULTS")
        print("=" * 100)

        total_scenarios = self.results[
            "total_scenarios"
        ]

        passed_scenarios = self.results[
            "passed"
        ]

        failed_scenarios = self.results[
            "failed"
        ]

        total_turns = self.results[
            "total_turns"
        ]

        passed_turns = self.results[
            "passed_turns"
        ]

        failed_turns = self.results[
            "failed_turns"
        ]

        scenario_rate = (
            passed_scenarios
            / total_scenarios
            * 100
            if total_scenarios
            else 0
        )

        turn_rate = (
            passed_turns
            / total_turns
            * 100
            if total_turns
            else 0
        )

        print(
            f"\nScenarios:"
        )

        print(
            f"  Total:   {total_scenarios}"
        )

        print(
            f"  Passed:  {passed_scenarios}"
        )

        print(
            f"  Failed:  {failed_scenarios}"
        )

        print(
            f"  Rate:    {scenario_rate:.1f}%"
        )

        print(
            f"\nTurns:"
        )

        print(
            f"  Total:   {total_turns}"
        )

        print(
            f"  Passed:  {passed_turns}"
        )

        print(
            f"  Failed:  {failed_turns}"
        )

        print(
            f"  Rate:    {turn_rate:.1f}%"
        )

        print(
            "\nScenario Breakdown:"
        )

        print("-" * 100)

        for scenario in self.results[
            "scenarios"
        ]:
            status = (
                "[PASS]"
                if scenario["passed"]
                else "[FAIL]"
            )

            print(
                f"{status} "
                f"{scenario['name']}"
            )

            for turn in scenario["turns"]:
                turn_status = (
                    "[PASS]"
                    if turn["passed"]
                    else "[FAIL]"
                )

                print(
                    f"   {turn_status} "
                    f"Turn {turn['turn']} "
                    f"({turn['type']})"
                )

                if not turn["passed"]:
                    print(
                        f"      "
                        f"{turn.get('error', 'Unknown error')}"
                    )

        output_file = "eval_results.json"

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                self.results,
                f,
                indent=2,
                ensure_ascii=False
            )

        print(
            f"\nDetailed results saved to: "
            f"{output_file}"
        )

        print(
            "\n" + "=" * 100
        )

        if (
            failed_scenarios == 0
            and failed_turns == 0
        ):
            print(
                "OVERALL RESULT: [PASS]"
            )
        else:
            print(
                "OVERALL RESULT: [FAIL]"
            )

        print(
            "=" * 100
        )


async def main():
    harness = EvaluationHarness()

    try:
        await harness.run_all_scenarios()

    except KeyboardInterrupt:
        print(
            "\nEvaluation interrupted."
        )

    except Exception as e:
        print(
            f"\nEvaluation failed: {e}"
        )
        raise


if __name__ == "__main__":
    asyncio.run(main())