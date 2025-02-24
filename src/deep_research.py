from typing import Callable, List, TypeVar, Any
import asyncio
import datetime
import json
import os

import math

from dotenv import load_dotenv
from google.genai import types

import google.generativeai as genai

from google import genai as genai_client

from google.ai.generativelanguage_v1beta.types import content


class ResearchProgress:
    def __init__(self, depth: int, breadth: int):
        self.total_depth = depth
        self.total_breadth = breadth
        self.current_depth = depth
        self.current_breadth = 0
        self.queries_by_depth = {}
        self.query_order = []  # Track order of queries
        self.query_parents = {}  # Track parent-child relationships
        self.total_queries = 0
        self.completed_queries = 0

    def start_query(self, query: str, depth: int, parent_query: str = None):
        """Record the start of a new query"""
        if depth not in self.queries_by_depth:
            self.queries_by_depth[depth] = {}

        if query not in self.queries_by_depth[depth]:
            self.queries_by_depth[depth][query] = {
                "completed": False,
                "learnings": [],
                "timestamp": len(self.query_order)  # Use position as timestamp
            }
            self.query_order.append(query)
            if parent_query:
                self.query_parents[query] = parent_query
            self.total_queries += 1

        self.current_depth = depth
        self.current_breadth = len(self.queries_by_depth[depth])
        self._report_progress(f"Starting query: {query}")

    def add_learning(self, query: str, depth: int, learning: str):
        """Record a learning for a specific query"""
        if depth in self.queries_by_depth and query in self.queries_by_depth[depth]:
            self.queries_by_depth[depth][query]["learnings"].append(learning)
            self._report_progress(f"Added learning for query: {query}")

    def complete_query(self, query: str, depth: int):
        """Mark a query as completed"""
        if depth in self.queries_by_depth and query in self.queries_by_depth[depth]:
            self.queries_by_depth[depth][query]["completed"] = True
            self.completed_queries += 1
            self._report_progress(f"Completed query: {query}")

    def _report_progress(self, action: str):
        """Report current progress"""
        print(f"\nResearch Progress Update:")
        print(f"Action: {action}")
        print(f"Overall Progress: {self.completed_queries}/{self.total_queries} queries completed")

        print("\nResearch Tree:")
        
        def print_subtree(query, indent=""):
            data = None
            depth = None
            # Find the query data
            for d, queries in self.queries_by_depth.items():
                if query in queries:
                    data = queries[query]
                    depth = d
                    break
            
            if data:
                status = "✓" if data["completed"] else "⋯"
                print(f"{indent}├── [{status}] {query}")
                if data["learnings"]:
                    print(f"{indent}│   └── {len(data['learnings'])} learnings")
                
                # Find and print child queries
                children = [q for q, p in self.query_parents.items() if p == query]
                for child in children:
                    print_subtree(child, indent + "    ")

        # Find root queries (those without parents)
        root_queries = [q for q in self.query_order if q not in self.query_parents]
        
        # Print tree starting from each root
        for query in root_queries:
            print_subtree(query)

        print(f"\nTotal Progress: {self.completed_queries}/{self.total_queries} queries completed")
        print("")

    def get_learnings_by_query(self):
        """Get all learnings organized by query"""
        learnings = {}
        for depth, queries in self.queries_by_depth.items():
            for query, data in queries.items():
                if data["learnings"]:
                    learnings[query] = data["learnings"]
        return learnings


load_dotenv()


T = TypeVar('T')


class AsyncLimit:
    def __init__(self, limit: int):
        self.sem = asyncio.Semaphore(limit)
        self.loop = asyncio.get_event_loop()

    async def run(self, fn: Callable[..., T], *args, **kwargs) -> T:
        async with self.sem:
            # Ensure the function runs in the correct event loop
            if asyncio.iscoroutinefunction(fn):
                return await fn(*args, **kwargs)
            else:
                return await self.loop.run_in_executor(None, fn, *args, **kwargs)

    async def map(self, fn: Callable[..., T], items: List[Any]) -> List[T]:
        # Create and await tasks properly
        tasks = []
        for item in items:
            task = await self.run(fn, item)
            tasks.append(task)
        return tasks


class DeepSearch:
    def __init__(self, api_key: str, mode: str = "balanced"):
        """
        Initialize DeepSearch with a mode parameter:
        - "fast": Prioritizes speed (reduced breadth/depth, highest concurrency)
        - "balanced": Default balance of speed and comprehensiveness
        - "comprehensive": Maximum detail and coverage
        """
        self.api_key = api_key
        self.async_limit = AsyncLimit(8)
        self.model_name = "gemini-2.0-flash"
        self.query_history = set()
        self.mode = mode
        genai.configure(api_key=self.api_key)

    def determine_research_breadth_and_depth(self, query: str):
        user_prompt = f"""
		You are a research planning assistant. Your task is to determine the appropriate breadth and depth for researching a topic defined by a user's query. Evaluate the query's complexity and scope, then recommend values on the following scales:

		Breadth: Scale of 1 (very narrow) to 10 (extensive, multidisciplinary).
		Depth: Scale of 1 (basic overview) to 5 (highly detailed, in-depth analysis).
		Defaults:

		Breadth: 4
		Depth: 2
		Note: More complex or "harder" questions should prompt higher ratings on one or both scales, reflecting the need for more extensive research and deeper analysis.

		Response Format:
		Output your recommendation in JSON format, including an explanation. For example:
		```json
		{{
			"breadth": 4,
			"depth": 2,
			"explanation": "The topic is moderately complex; a broad review is needed (breadth 4) with a basic depth analysis (depth 2)."
		}}
		```

		Here is the user's query:
		<query>{query}</query>
		"""

        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
            "response_schema": content.Schema(
                type=content.Type.OBJECT,
                enum=[],
                required=["breadth", "depth", "explanation"],
                properties={
                    "breadth": content.Schema(type=content.Type.NUMBER),
                    "depth": content.Schema(type=content.Type.NUMBER),
                    "explanation": content.Schema(type=content.Type.STRING),
                },
            ),
        }

        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=generation_config,
        )

        response = model.generate_content(user_prompt)
        answer = response.text

        return json.loads(answer)

    def generate_follow_up_questions(
        query: str,
        max_questions: int = 3,
    ):
        user_prompt = f"""
		Given the following query from the user, ask some follow up questions to clarify the research direction.

		Return a maximum of {max_questions} questions, but feel free to return less if the original query is clear: <query>{query}</query>
		"""

        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
            "response_schema": content.Schema(
                type=content.Type.OBJECT,
                enum=[],
                required=["follow_up_queries"],
                properties={
                    "follow_up_queries": content.Schema(
                        type=content.Type.ARRAY,
                        items=content.Schema(
                            type=content.Type.STRING,
                        ),
                    ),
                },
            ),
        }

        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=generation_config,
        )

        response = model.generate_content(user_prompt)
        answer = response.text

        return json.loads(answer)["follow_up_queries"]

    def generate_queries(
            self,
            query: str,
            num_queries: int = 3,
            learnings: list[str] = [],
            previous_queries: set[str] = None  # Add previous_queries parameter
    ):
        now = datetime.datetime.now().strftime("%Y-%m-%d")

        # Format previous queries for the prompt
        previous_queries_text = ""
        if previous_queries:
            previous_queries_text = "\n\nPreviously asked queries (avoid generating similar ones):\n" + \
                "\n".join([f"- {q}" for q in previous_queries])

        user_prompt = f"""
        Given the following prompt from the user, generate a list of SERP queries to research the topic. Return a maximum
        of {num_queries} queries, but feel free to return less if the original prompt is clear.

        IMPORTANT: Each query must be unique and significantly different from both each other AND the previously asked queries.
        Avoid semantic duplicates or queries that would likely return similar information.

        Original prompt: <prompt>${query}</prompt>
        {previous_queries_text}
        """

        learnings_prompt = "" if not learnings else "Here are some learnings from previous research, use them to generate more specific queries: " + \
            "\n".join(learnings)

        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_schema": content.Schema(
                type=content.Type.OBJECT,
                enum=[],
                required=["queries"],
                properties={
                    "queries": content.Schema(
                        type=content.Type.ARRAY,
                        items=content.Schema(
                            type=content.Type.STRING,
                        ),
                    ),
                },
            ),
            "response_mime_type": "application/json",
        }

        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=generation_config,
        )

        # generate a list of queries
        response = model.generate_content(
            user_prompt + learnings_prompt
        )

        answer = response.text

        answer_list = json.loads(answer)["queries"]

        return answer_list

    def format_text_with_sources(self, response_dict: dict, answer: str):
        """
        Format text with sources from Gemini response, adding citations at specified positions.
        Returns tuple of (formatted_text, sources_dict).
        """
        if not response_dict or not response_dict.get('candidates'):
            return answer, {}

        # Get grounding metadata from the response
        grounding_metadata = response_dict['candidates'][0].get(
            'grounding_metadata')
        if not grounding_metadata:
            return answer, {}

        # Get grounding chunks and supports
        grounding_chunks = grounding_metadata.get('grounding_chunks', [])
        grounding_supports = grounding_metadata.get('grounding_supports', [])

        if not grounding_chunks or not grounding_supports:
            return answer, {}

        try:
            # Create mapping of URLs
            sources = {
                i: {
                    'link': chunk.get('web', {}).get('uri', ''),
                    'title': chunk.get('web', {}).get('title', '')
                }
                for i, chunk in enumerate(grounding_chunks)
                if chunk.get('web')
            }

            # Create a list of (position, citation) tuples
            citations = []
            for support in grounding_supports:
                segment = support.get('segment', {})
                indices = support.get('grounding_chunk_indices', [])

                if indices and segment and segment.get('end_index') is not None:
                    end_index = segment['end_index']
                    source_idx = indices[0]
                    if source_idx in sources:
                        citation = f"[[{source_idx + 1}]]({sources[source_idx]['link']})"
                        citations.append((end_index, citation))

            # Sort citations by position (end_index)
            citations.sort(key=lambda x: x[0])

            # Insert citations into the text
            result = ""
            last_pos = 0
            for pos, citation in citations:
                result += answer[last_pos:pos]
                result += citation
                last_pos = pos

            # Add any remaining text
            result += answer[last_pos:]

            return result, sources

        except Exception as e:
            print(f"Error processing grounding metadata: {e}")
            return answer, {}

    def search(self, query: str):
        client = genai_client.Client(
            api_key=os.environ.get("GEMINI_KEY")
        )

        model_id = "gemini-2.0-flash"

        google_search_tool = types.Tool(
            google_search=types.GoogleSearch()
        )

        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
            "response_modalities": ["TEXT"],
            "tools": [google_search_tool]
        }

        response = client.models.generate_content(
            model=model_id,
            contents=query,
            config=generation_config
        )

        response_dict = response.model_dump()

        formatted_text, sources = self.format_text_with_sources(
            response_dict, response.text)

        return formatted_text, sources

    async def process_result(
        self,
        query: str,
        result: str,
        num_learnings: int = 3,
        num_follow_up_questions: int = 3,
    ):
        print(f"Processing result for query: {query}")

        user_prompt = f"""
		Given the following result from a SERP search for the query <query>{query}</query>, generate a list of learnings from the result. Return a maximum of {num_learnings} learnings, but feel free to return less if the result is clear. Make sure each learning is unique and not similar to each other. The learnings should be concise and to the point, as detailed and information dense as possible. Make sure to include any entities like people, places, companies, products, things, etc in the learnings, as well as any exact metrics, numbers, or dates. The learnings will be used to research the topic further.
		"""

        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
            "response_schema": content.Schema(
                type=content.Type.OBJECT,
                enum=[],
                required=["learnings", "follow_up_questions"],
                properties={
                    "learnings": content.Schema(
                        type=content.Type.ARRAY,
                        items=content.Schema(
                            type=content.Type.STRING
                        )
                    ),
                    "follow_up_questions": content.Schema(
                        type=content.Type.ARRAY,
                        items=content.Schema(
                            type=content.Type.STRING
                        )
                    )
                },
            ),
        }

        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=generation_config,
        )

        response = model.generate_content(user_prompt)
        answer = response.text

        answer_json = json.loads(answer)

        learnings = answer_json["learnings"]
        follow_up_questions = answer_json["follow_up_questions"]

        print(f"Results from {query}:")
        print(f"Learnings: {learnings}\n")
        print(f"Follow up questions: {follow_up_questions}\n")

        return answer_json

    def _are_queries_similar(self, query1: str, query2: str) -> bool:
        """Helper method to check if two queries are semantically similar using Gemini"""
        user_prompt = f"""
        Compare these two search queries and determine if they are semantically similar 
        (i.e., would likely return similar search results or are asking about the same topic):

        Query 1: {query1}
        Query 2: {query2}

        Consider:
        1. Key concepts and entities
        2. Intent of the queries
        3. Scope and specificity
        4. Core topic overlap

        Only respond with true if the queries are notably similar, false otherwise.
        """

        generation_config = {
            "temperature": 0.1,  # Low temperature for more consistent results
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
            "response_schema": content.Schema(
                type=content.Type.OBJECT,
                required=["are_similar"],
                properties={
                    "are_similar": content.Schema(
                        type=content.Type.BOOLEAN,
                        description="True if queries are semantically similar, false otherwise"
                    )
                }
            )
        }

        try:
            model = genai.GenerativeModel(
                "gemini-2.0-flash",
                generation_config=generation_config,
            )

            response = model.generate_content(user_prompt)
            answer = json.loads(response.text)
            return answer["are_similar"]
        except Exception as e:
            print(f"Error comparing queries: {str(e)}")
            # In case of error, assume queries are different to avoid missing potentially unique results
            return False

    async def deep_research(self, query: str, breadth: int, depth: int, learnings: list[str] = [], visited_urls: dict[int, dict] = {}, parent_query: str = None):
        progress = ResearchProgress(depth, breadth)

        # Adjust number of queries based on mode
        max_queries = {
            "fast": 3,
            "balanced": 7,
            "comprehensive": 5 # kept lower than balanced due to recursive multiplication
        }[self.mode]

        queries = self.generate_queries(
            query,
            min(breadth, max_queries),
            learnings,
            previous_queries=self.query_history
        )

        self.query_history.update(queries)
        unique_queries = list(queries)[:breadth]

        async def process_query(query_str: str, parent_query: str = None):
            try:
                progress.start_query(query_str, depth, parent_query)

                result = self.search(query_str)
                processed_result = await self.process_result(
                    query=query_str,
                    result=result[0],
                    num_learnings=min(3, math.ceil(breadth / 2)),
                    num_follow_up_questions=min(2, math.ceil(breadth / 2))
                )

                # Record learnings
                for learning in processed_result["learnings"]:
                    progress.add_learning(query_str, depth, learning)

                new_urls = result[1]
                max_idx = max(visited_urls.keys()) if visited_urls else -1
                all_urls = {
                    **visited_urls,
                    **{(i + max_idx + 1): url_data for i, url_data in new_urls.items()}
                }

                # Only go deeper if in comprehensive mode and depth > 1
                if self.mode == "comprehensive" and depth > 1:
                    # Reduced breadth for deeper levels
                    new_breadth = min(2, math.ceil(breadth / 2))
                    new_depth = depth - 1

                    # Select most important follow-up question instead of using all
                    if processed_result['follow_up_questions']:
                        # Take only the most relevant question
                        next_query = processed_result['follow_up_questions'][0]

                        sub_results = await self.deep_research(
                            query=next_query,
                            breadth=new_breadth,
                            depth=new_depth,
                            learnings=learnings +
                            processed_result["learnings"],
                            visited_urls=all_urls,
                            parent_query=query_str  # Pass current query as parent
                        )

                        progress.complete_query(query_str, depth)
                        return sub_results

                progress.complete_query(query_str, depth)
                return {
                    "learnings": learnings + processed_result["learnings"],
                    "visited_urls": all_urls
                }

            except Exception as e:
                print(f"Error processing query {query_str}: {str(e)}")
                progress.complete_query(query_str, depth)
                return {
                    "learnings": [],
                    "visited_urls": {}
                }

        # Process queries concurrently
        results = await self.async_limit.map(process_query, unique_queries)

        # Combine results
        all_learnings = list(set(
            learning
            for result in results
            for learning in result["learnings"]
        ))

        all_urls = {}
        current_idx = 0
        seen_urls = set()
        for result in results:
            for url_data in result["visited_urls"].values():
                if url_data['link'] not in seen_urls:
                    all_urls[current_idx] = url_data
                    seen_urls.add(url_data['link'])
                    current_idx += 1

        return {
            "learnings": all_learnings,
            "visited_urls": all_urls
        }

    def generate_final_report(self, query: str, learnings: list[str], visited_urls: dict[int, dict]) -> str:
        # Format sources and learnings for the prompt
        sources_text = "\n".join([
            f"- {data['title']}: {data['link']}"
            for data in visited_urls.values()
        ])
        learnings_text = "\n".join([f"- {learning}" for learning in learnings])

        user_prompt = f"""
        You are a creative research analyst tasked with synthesizing findings into an engaging and informative report.
        Create a comprehensive research report (minimum 3000 words) based on the following query and findings.
        
        Original Query: {query}
        
        Key Findings:
        {learnings_text}
        
        Sources Used:
        {sources_text}
        
        Guidelines:
        1. Design a creative and engaging report structure that best fits the content and topic
        2. Feel free to use any combination of:
           - Storytelling elements
           - Case studies
           - Scenarios
           - Visual descriptions
           - Analogies and metaphors
           - Creative section headings
           - Thought experiments
           - Future projections
           - Historical parallels
        3. Make the report engaging while maintaining professionalism
        4. Include all relevant data points but present them in an interesting way
        5. Structure the information in whatever way makes the most logical sense for this specific topic
        6. Feel free to break conventional report formats if a different approach would be more effective
        7. Consider using creative elements like:
           - "What if" scenarios
           - Day-in-the-life examples
           - Before/After comparisons
           - Expert perspectives
           - Trend timelines
           - Problem-solution frameworks
           - Impact matrices
        
        Requirements:
        - Minimum 3000 words
        - Must include all key findings and data points
        - Must maintain factual accuracy
        - Must be well-organized and easy to follow
        - Must include clear conclusions and insights
        - Must cite sources appropriately
        
        Be bold and creative in your approach while ensuring the report effectively communicates all the important information!
        """

        generation_config = {
            "temperature": 0.9,  # Increased for more creativity
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }

        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=generation_config,
        )

        response = model.generate_content(user_prompt)

        # Format the response with inline citations
        formatted_text, sources = self.format_text_with_sources(
            response.to_dict(),
            response.text
        )

        # Add sources section
        sources_section = "\n# Sources\n" + "\n".join([
            f"- [{data['title']}]({data['link']})"
            for data in visited_urls.values()
        ])

        return formatted_text + sources_section



