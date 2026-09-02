import asyncio
import sys
import os
import json
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import config
from core.chat_loop import ChatLoop
from memory.db import MemoryDB
from eval.test_scenarios import TEST_SCENARIOS

class EvaluationHarness:
    def __init__(self):
        self.chat_loop = ChatLoop(config)
        self.memory_db = MemoryDB(config.db_path)
        self.results = {
            "total_scenarios": 0,
            "passed": 0,
            "failed": 0,
            "scenarios": []
        }
    
    async def run_all_scenarios(self):
        """Run all test scenarios"""
        print("\n" + "="*80)
        print("AI COMPANION EVALUATION HARNESS")
        print("="*80 + "\n")
        
        await self.chat_loop.initialize()
        
        for scenario in TEST_SCENARIOS:
            await self.run_scenario(scenario)
        
        self.print_results()
    
    async def run_scenario(self, scenario):
        """Run a single test scenario"""
        scenario_name = scenario["name"]
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'='*60}")
        
        scenario_result = {
            "name": scenario_name,
            "turns": [],
            "passed": True,
            "failures": []
        }
        
        # Clear DB for fresh test
        self.memory_db = MemoryDB(config.db_path)
        self.chat_loop = ChatLoop(config)
        await self.chat_loop.initialize()
        
        for i, turn in enumerate(scenario["turns"], 1):
            print(f"\n[Turn {i}]")
            user_msg = turn["user"]
            print(f"User: {user_msg}")
            
            # Get response
            response = await self.chat_loop.get_response(user_msg)
            print(f"Alex: {response}\n")
            
            check_type = turn.get("check_type", "chat")
            turn_result = {"turn": i, "type": check_type, "passed": True}
            
            # Check based on type
            if check_type == "store":
                turn_result = await self._check_facts_stored(turn, turn_result)
            elif check_type == "recall":
                turn_result = await self._check_facts_recalled(response, turn, turn_result)
            elif check_type == "update":
                turn_result = await self._check_facts_updated(turn, turn_result)
            elif check_type == "personality":
                turn_result = await self._check_personality(response, turn, turn_result)
            elif check_type == "chat":
                print(f"[Chat turn - no validation]")
            
            scenario_result["turns"].append(turn_result)
            
            if not turn_result["passed"]:
                scenario_result["passed"] = False
                scenario_result["failures"].append(turn_result)
        
        # Update overall results
        self.results["total_scenarios"] += 1
        if scenario_result["passed"]:
            self.results["passed"] += 1
        else:
            self.results["failed"] += 1
        
        self.results["scenarios"].append(scenario_result)
        
        status = "[PASS]" if scenario_result["passed"] else "[FAIL]"
        print(f"\n{status}")
    
    async def _check_facts_stored(self, turn, result):
        """Check if facts were stored"""
        expected_facts = turn.get("expected_facts", [])
        all_facts = await self.memory_db.get_all_facts()
        stored_texts = [f['content'].lower() for f in all_facts]
        
        missing = []
        for expected in expected_facts:
            if not any(expected.lower() in stored for stored in stored_texts):
                missing.append(expected)
        
        if missing:
            result["passed"] = False
            result["error"] = f"Missing facts: {missing}"
            print(f"[FAIL] Expected {expected_facts}, got {stored_texts}")
        else:
            print(f"[PASS] All facts stored")
        
        return result
    
    async def _check_facts_recalled(self, response, turn, result):
        """Check if facts were recalled in response"""
        expected_recall = turn.get("expected_recall", [])
        not_recall = turn.get("not_recall", [])
        response_lower = response.lower()
        
        # Debug: show what's in the DB
        all_facts = await self.memory_db.get_all_facts()
        print(f"\n[DB State] Total facts: {len(all_facts)}")
        for fact in all_facts:
            print(f"  - {fact['content']}")
        
        missing = []
        for expected in expected_recall:
            if expected.lower() not in response_lower:
                missing.append(expected)
        
        wrongly_recalled = []
        for should_not in not_recall:
            if should_not.lower() in response_lower:
                wrongly_recalled.append(should_not)
        
        if missing or wrongly_recalled:
            result["passed"] = False
            errors = []
            if missing:
                errors.append(f"Missing recall: {missing}")
            if wrongly_recalled:
                errors.append(f"Wrong recall: {wrongly_recalled}")
            result["error"] = ", ".join(errors)
            print(f"[FAIL] {result['error']}")
            print(f"Response: {response}")
        else:
            print(f"[PASS] All facts recalled correctly")
        
        return result
    
    async def _check_facts_updated(self, turn, result):
        """Check if old facts were updated, not duplicated"""
        all_facts = await self.memory_db.get_all_facts()
        
        print(f"\n[DB State] Total facts: {len(all_facts)}")
        for fact in all_facts:
            print(f"  - {fact['content']}")
        
        # Count facts in category
        fact_count = len([f for f in all_facts if f.get('category') == turn.get("category", "work")])
        
        if fact_count > 1:
            result["passed"] = False
            result["error"] = f"Facts duplicated instead of updated. Count: {fact_count}"
            print(f"[FAIL] Facts duplicated")
        else:
            print(f"[PASS] Fact updated correctly")
        
        return result
    
    async def _check_personality(self, response, turn, result):
        """Check if personality traits are consistent"""
        expected_traits = turn.get("expected_personality", [])
        not_traits = turn.get("not_personality", [])
        response_lower = response.lower()
        
        missing = []
        for trait in expected_traits:
            if trait.lower() not in response_lower:
                missing.append(trait)
        
        contradicted = []
        for trait in not_traits:
            if trait.lower() in response_lower:
                contradicted.append(trait)
        
        if missing or contradicted:
            result["passed"] = False
            errors = []
            if missing:
                errors.append(f"Missing traits: {missing}")
            if contradicted:
                errors.append(f"Contradicted traits: {contradicted}")
            result["error"] = ", ".join(errors)
            print(f"[FAIL] {result['error']}")
        else:
            print(f"[PASS] Personality consistent")
        
        return result
    
    def print_results(self):
        """Print evaluation results"""
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        
        total = self.results["total_scenarios"]
        passed = self.results["passed"]
        failed = self.results["failed"]
        pass_rate = (passed / total * 100) if total > 0 else 0
        
        print(f"\nTotal Scenarios: {total}")
        print(f"Passed: {passed} [PASS]")
        print(f"Failed: {failed} [FAIL]")
        print(f"Pass Rate: {pass_rate:.1f}%\n")
        
        print("Scenario Breakdown:")
        print("-" * 80)
        
        for scenario in self.results["scenarios"]:
            status = "[PASS]" if scenario["passed"] else "[FAIL]"
            print(f"{status} {scenario['name']}")
            
            if scenario["failures"]:
                for failure in scenario["failures"]:
                    print(f"   Turn {failure['turn']}: {failure.get('error', 'Unknown error')}")
        
        # Save results to file
        output_file = "eval_results.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")

async def main():
    harness = EvaluationHarness()
    await harness.run_all_scenarios()

if __name__ == "__main__":
    asyncio.run(main())