import argparse
import asyncio
import os
import time

from src.deep_research import DeepSearch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run deep search queries')
    parser.add_argument('query', type=str, help='The search query')
    parser.add_argument('--mode', type=str, choices=['fast', 'balanced', 'comprehensive'],
                        default='balanced', help='Research mode (default: balanced)')
    parser.add_argument('--num-queries', type=int, default=3,
                        help='Number of queries to generate (default: 3)')
    parser.add_argument('--learnings', nargs='*', default=[],
                        help='List of previous learnings')

    args = parser.parse_args()

    # Start the timer
    start_time = time.time()

    # Get API key from environment variable
    api_key = os.getenv('GEMINI_KEY')
    if not api_key:
        raise ValueError("Please set GEMINI_KEY environment variable")

    deep_search = DeepSearch(api_key, mode=args.mode)

    breadth_and_depth = deep_search.determine_research_breadth_and_depth(
        args.query)

    breadth = breadth_and_depth["breadth"]
    depth = breadth_and_depth["depth"]
    explanation = breadth_and_depth["explanation"]

    print(f"Breadth: {breadth}")
    print(f"Depth: {depth}")
    print(f"Explanation: {explanation}")

    print("To better understand your research needs, please answer these follow-up questions:")

    follow_up_questions = deep_search.generate_follow_up_questions(args.query)

    # get answers to the follow up questions
    answers = []
    for question in follow_up_questions:
        answer = input(f"{question}: ")
        answers.append({
            "question": question,
            "answer": answer
        })

    questions_and_answers = "\n".join(
        [f"{answer['question']}: {answer['answer']}" for answer in answers])

    combined_query = f"Initial query: {args.query}\n\n Follow up questions and answers: {questions_and_answers}"

    print(f"\nHere is the combined query: {combined_query}\n\n")

    print("Starting research... \n")

    # Run the deep research
    results = asyncio.run(deep_search.deep_research(
        query=combined_query,
        breadth=breadth,
        depth=depth,
        learnings=[],
        visited_urls={}
    ))

    # Generate and print the final report
    final_report = asyncio.run(deep_search.generate_final_report(
        query=combined_query,
        learnings=results["learnings"],
        visited_urls=results["visited_urls"]
    ))

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    print("\nFinal Research Report:")
    print("=====================")
    print(final_report)
    print(f"\nTotal research time: {minutes} minutes and {seconds} seconds")

    # Save the report to a file
    with open("final_report.md", "w") as f:
        f.write(final_report)
        f.write(
            f"\n\nTotal research time: {minutes} minutes and {seconds} seconds")
            