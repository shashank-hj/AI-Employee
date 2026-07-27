import sys
import time
import httpx

ORCHESTRATOR_URL = "http://127.0.0.1:8001"


def display_response(response_data: dict, duration_ms: float):
    print()
    print("─" * 60)
    print(f"  Agent ({response_data.get('request_id', 'N/A')})  |  {duration_ms:.0f}ms")
    print("─" * 60)

    for step in response_data.get("steps", []):
        step_num = step.get("step_index", "?")
        tool_name = step.get("tool_name", "unknown")
        result = step.get("result") or {}

        print(f"\n  Step {step_num}: {tool_name}")
        if result.get("success"):
            data = result.get("data", {})
            if data:
                for k, v in data.items():
                    print(f"    {k}: {v}")
        else:
            print(f"    ERROR: {result.get('error', 'unknown error')}")

    print()
    print(f"  {response_data.get('final_response', '')}")
    print("─" * 60)
    print()


def main():
    print("=" * 60)
    print("  AI Employee Platform - Interactive CLI")
    print("  Type 'exit' or 'quit' to stop")
    print("=" * 60)
    print()

    client = httpx.Client(timeout=60.0)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not user_input:
            continue

        print("\nOrchestrator: thinking...", end="\r")

        try:
            start = time.time()
            resp = client.post(
                f"{ORCHESTRATOR_URL}/api/agent/run",
                json={"user_input": user_input},
            )
            elapsed = (time.time() - start) * 1000

            if resp.status_code == 200:
                display_response(resp.json(), elapsed)
            elif resp.status_code == 422:
                data = resp.json()
                print(f"\n  Validation error: {data.get('detail', data)}")
            else:
                print(f"\n  Error [{resp.status_code}]: {resp.text}")

        except httpx.ConnectError:
            print("\n  Cannot connect to orchestrator at http://127.0.0.1:8001")
            print("  Make sure the orchestrator is running first.")
            break
        except Exception as e:
            print(f"\n  Unexpected error: {e}")

    client.close()


if __name__ == "__main__":
    main()
