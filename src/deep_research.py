import asyncio
import datetime
import json
import os
from typing import List
import uuid
import argparse
import time
import math

from dotenv import load_dotenv

from google import genai
from google.genai import types
from pydantic import BaseModel


class ResearchProgress:
    def __init__(self, depth: int, breadth: int):
        self.total_depth = depth
        self.total_breadth = breadth
        self.current_depth = depth
        self.current_breadth = 0
        self.queries_by_depth = {}
        self.query_order = []  # Track order of queries
        self.query_parents = {}  # Track parent-child relationships
        self.total_queries = 0  # Total number of queries including sub-queries
        self.completed_queries = 0
        self.query_ids = {}  # Store persistent IDs for queries
        self.root_query = None  # Store the root query

    async def start_query(self, query: str, depth: int, parent_query: str = None):
        """Record the start of a new query"""
        # Generate a unique ID for this query
        query_id = str(uuid.uuid4())
        self.query_ids[query] = query_id

        # If this is the first query, set it as the root
        if self.root_query is None:
            self.root_query = query

        # Initialize the depth level if it doesn't exist
        if depth not in self.queries_by_depth:
            self.queries_by_depth[depth] = {}

        # Add the query to the appropriate depth level if it's not already there
        if query not in self.queries_by_depth[depth]:
            self.queries_by_depth[depth][query] = {
                "completed": False,
                "learnings": [],
                "sources": [],  # Add sources list to store source information
                "id": self.query_ids[query]  # Use persistent ID
            }
            self.query_order.append(query)
            if parent_query:
                self.query_parents[query] = parent_query
            self.total_queries += 1

        self.current_depth = depth
        self.current_breadth = len(self.queries_by_depth[depth])
        await self._report_progress("query_started")

    async def add_learning(self, query: str, depth: int, learning: str):
        if depth in self.queries_by_depth and query in self.queries_by_depth[depth]:
            self.queries_by_depth[depth][query]["learnings"].append(
                learning)
            await self._report_progress("learning_added")

    async def complete_query(self, query: str, depth: int):
        """Mark a query as completed"""
        if depth in self.queries_by_depth and query in self.queries_by_depth[depth]:
            if not self.queries_by_depth[depth][query]["completed"]:
                self.queries_by_depth[depth][query]["completed"] = True
                self.completed_queries += 1
                await self._report_progress(f"Completed query: {query}")

                # Check if parent query exists and update its status if all children are complete
                parent_query = self.query_parents.get(query)
                if parent_query:
                    await self._update_parent_status(parent_query)

    async def add_sources(self, query: str, depth: int, sources: list[dict[str, str]]):
        """Record sources for a specific query"""
        if depth in self.queries_by_depth and query in self.queries_by_depth[depth]:
            # Add new sources that aren't already in the list
            current_sources = self.queries_by_depth[depth][query]["sources"]
            current_urls = {source["url"] for source in current_sources}

            for source in sources:
                if source["url"] not in current_urls:
                    current_sources.append(source)
                    current_urls.add(source["url"])

            await self._report_progress(f"Added sources for query: {query}")

    async def _update_parent_status(self, parent_query: str):
        """Update parent query status based on children completion"""
        # Find all children of this parent
        children = [q for q, p in self.query_parents.items() if p ==
                    parent_query]

        # Check if all children are complete
        parent_depth = next((d for d, queries in self.queries_by_depth.items()
                             if parent_query in queries), None)

        if parent_depth is not None:
            all_children_complete = all(
                self.queries_by_depth[d][q]["completed"]
                for q in children
                for d in self.queries_by_depth
                if q in self.queries_by_depth[d]
            )

            if all_children_complete:
                # Complete the parent query
                await self.complete_query(parent_query, parent_depth)

    async def _report_progress(self, action: str):
        """Report current progress and stream to client if callback provided"""
        # Build event data for streaming
        progress_data = {
            "type": "research_progress",
            "action": action,
            "timestamp": datetime.datetime.now().isoformat(),
            "completed_queries": self.completed_queries,
            "total_queries": self.total_queries,
            "progress_percentage": int((self.completed_queries / max(1, self.total_queries)) * 100)
        }

        # Add tree structure if root query exists
        if self.root_query:
            progress_data["tree"] = self._build_research_tree()

        # Print progress to console
        print(
            f"[Progress] {action}: {progress_data['progress_percentage']}% complete")

    def _build_research_tree(self):
        """Build a tree structure of the research queries"""
        def build_node(query):
            """Recursively build the tree node"""
            # Find the depth for this query
            depth = next((d for d, queries in self.queries_by_depth.items()
                          if query in queries), 0)

            data = self.queries_by_depth[depth][query]

            # Find all children of this query
            children = [q for q, p in self.query_parents.items() if p == query]

            return {
                "query": query,
                "id": self.query_ids[query],
                "status": "completed" if data["completed"] else "in_progress",
                "depth": depth,
                "learnings": data["learnings"],
                "sources": data["sources"],  # Include sources in the tree
                "sub_queries": [build_node(child) for child in children],
                "parent_query": self.query_parents.get(query)
            }

        # Start building from the root query
        if self.root_query:
            return build_node(self.root_query)
        return {}

    def get_learnings_by_query(self):
        """Get all learnings organized by query"""
        learnings = {}
        for depth, queries in self.queries_by_depth.items():
            for query, data in queries.items():
                if data["learnings"]:
                    learnings[query] = data["learnings"]
        return learnings


load_dotenv()


class DeepSearch:
    def __init__(self, api_key: str, mode: str = "balanced"):
        """
        Initialize DeepSearch with a mode parameter:
        - "fast": Prioritizes speed (reduced breadth/depth, highest concurrency)
        - "balanced": Default balance of speed and comprehensiveness
        - "comprehensive": Maximum detail and coverage
        """
        self.api_key = api_key
        self.model_name = "gemini-2.0-flash"
        self.query_history = set()
        self.mode = mode
        self.client = genai.Client(api_key=self.api_key)

    def determine_research_breadth_and_depth(self, query: str):
        """Determine the appropriate research breadth and depth based on the query complexity"""
        class ResearchParameters(BaseModel):
            breadth: int
            depth: int
            explanation: str

        user_prompt = f"""
        Analyze this research query and determine the appropriate breadth (number of parallel search queries) 
        and depth (levels of follow-up questions) needed for thorough research:

        Query: {query}

        Consider:
        1. Complexity of the topic
        2. Breadth of knowledge required
        3. Depth of expertise needed
        4. Potential for follow-up exploration

        Return a JSON object with:
        - "breadth": integer between 1-10 (number of parallel search queries)
        - "depth": integer between 1-5 (levels of follow-up questions)
        - "explanation": brief explanation of your reasoning
        """

        generation_config = {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
            "response_mime_type": "application/json",
            "response_schema": ResearchParameters,
        }

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=user_prompt,
                config=generation_config
            )

            # Get the parsed response using the Pydantic model
            parsed_response = response.parsed

            return {
                "breadth": parsed_response.breadth,
                "depth": parsed_response.depth,
                "explanation": parsed_response.explanation
            }

        except Exception as e:
            print(f"Error determining research parameters: {str(e)}")
            # Default values based on mode
            defaults = {
                "fast": {"breadth": 3, "depth": 1},
                "balanced": {"breadth": 5, "depth": 2},
                "comprehensive": {"breadth": 7, "depth": 3}
            }
            return defaults.get(self.mode, {"breadth": 5, "depth": 2, "explanation": "Using default values."})

    def generate_follow_up_questions(
        self,
        query: str,
        max_questions: int = 3,
    ):
        """Generate follow-up questions based on the initial query"""
        class FollowUpQuestions(BaseModel):
            follow_up_queries: list[str]

        user_prompt = f"""
        Based on the following user query, generate {max_questions} follow-up questions that would help clarify what the user wants to know.
		These questions should:
		1. Seek to understand the user's specific information needs
		2. Clarify ambiguous terms or concepts in the original query
		3. Determine the scope or boundaries of what the user is looking for
		4. Identify the user's level of familiarity with the topic
		5. Uncover the user's purpose or goal for seeking this information

		User Query: {query}

		Format your response as a JSON object with a single key "follow_up_queries" containing an array of strings.
		Example:
		```json
		{{
			"follow_up_queries": [
				"Could you specify what aspects of electric vehicles you're most interested in learning about?",
				"Are you looking for information about a specific brand or type of electric vehicle?",
				"Would you like to know about the technical details, environmental impact, or consumer aspects?"
			]
		}}
		```
        """

        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
            "response_mime_type": "application/json",
            "response_schema": FollowUpQuestions,
        }

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=user_prompt,
                config=generation_config
            )

            try:
                # Get the parsed response using the Pydantic model
                parsed_response = response.parsed
                return parsed_response.follow_up_queries
            except Exception as e:
                print(f"Error parsing follow-up questions: {str(e)}")
                # Fallback to simple text parsing
                lines = response.text.strip().split('\n')
                questions = []
                for line in lines:
                    line = line.strip()
                    if line and '?' in line:
                        questions.append(line)
                return questions[:max_questions]

        except Exception as e:
            print(f"Error generating follow-up questions: {str(e)}")
            return [f"What are the key aspects of {query}?"]

    async def generate_queries(
            self,
            query: str,
            num_queries: int = 3,
            learnings: list[str] = [],
            previous_queries: set[str] = None  # Add previous_queries parameter
    ):
        """Generate search queries based on the initial query and learnings"""
        if previous_queries is None:
            previous_queries = set()

        # Adjust the prompt based on the mode
        prompt_by_mode = {
            "fast": "Generate concise, focused search queries",
            "balanced": "Generate balanced search queries that explore different aspects",
            "comprehensive": "Generate comprehensive search queries that deeply explore the topic"
        }

        mode_prompt = prompt_by_mode.get(self.mode, prompt_by_mode["balanced"])

        # Format learnings for the prompt
        learnings_text = "\n".join([f"- {learning}" for learning in learnings])
        learnings_section = f"\nBased on what we've learned so far:\n{learnings_text}" if learnings else ""

        # Format previous queries for the prompt
        previous_queries_text = "\n".join([f"- {q}" for q in previous_queries])
        previous_queries_section = f"\nPrevious search queries (avoid repeating these):\n{previous_queries_text}" if previous_queries else ""

        user_prompt = f"""
        You are a research assistant helping to explore the topic: "{query}"
        
        {mode_prompt}.
        {learnings_section}
        {previous_queries_section}
        
        Generate {num_queries} specific search queries that would help gather comprehensive information about this topic.
        Each query should focus on a different aspect or subtopic.
        Make the queries specific and well-formed for a search engine.
        
        Format your response as a JSON object with a "queries" field containing an array of query strings.
        """

        class QueryResponse(BaseModel):
            queries: list[str]

        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
            "response_mime_type": "application/json",
            "response_schema": QueryResponse,
        }

        try:
            response = await self.client.aio.models.generate_content(
                model="gemini-2.0-flash",
                contents=user_prompt,
                config=generation_config
            )

            # Parse the response
            try:
                # Get the parsed response using the Pydantic model
                parsed_response = response.parsed
                queries = set(parsed_response.queries)

                # Filter out any queries that are too similar to previous ones
                unique_queries = set()
                for q in queries:
                    is_similar = False
                    for prev_q in previous_queries:
                        if await self._are_queries_similar(q, prev_q):
                            is_similar = True
                            break

                    if not is_similar:
                        unique_queries.add(q)

                return unique_queries
            except Exception as e:
                print(f"Error parsing query response: {str(e)}")
                # Fallback to simple text parsing if JSON parsing fails
                lines = response.text.strip().split('\n')
                queries = set()
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('{') and not line.startswith('}'):
                        queries.add(line)
                return queries

        except Exception as e:
            print(f"Error generating queries: {str(e)}")
            # Fallback to basic queries if generation fails
            return {f"{query} - aspect {i+1}" for i in range(num_queries)}

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

    async def search(self, query: str):
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

        response = await self.client.aio.models.generate_content(
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
        """Process search results to extract learnings and generate follow-up questions"""
        class ProcessedResult(BaseModel):
            learnings: list[str]
            follow_up_questions: list[str]

        user_prompt = f"""
        Analyze the following search results for the query: "{query}"
        
        Search Results:
        {result}
        
        Please extract:
        1. The {num_learnings} most important learnings or insights from these results
        2. {num_follow_up_questions} follow-up questions that would help explore this topic further
        
        Format your response as a JSON object with:
        - "learnings": array of learning strings
        - "follow_up_questions": array of question strings
        """

        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
            "response_mime_type": "application/json",
            "response_schema": ProcessedResult,
        }

        try:
            response = await self.client.aio.models.generate_content(
                model="gemini-2.0-flash",
                contents=user_prompt,
                config=generation_config
            )

            try:
                # Get the parsed response using the Pydantic model
                parsed_response = response.parsed
                return {
                    "learnings": parsed_response.learnings,
                    "follow_up_questions": parsed_response.follow_up_questions
                }
            except Exception as e:
                print(f"Error parsing process_result: {str(e)}")
                # Fallback to generating follow-up questions separately
                follow_up_questions = self.generate_follow_up_questions(
                    query, num_follow_up_questions
                )

                # Extract some basic learnings from the result
                learnings = [
                    f"Information about {query}",
                    f"Details related to {query}"
                ]

                return {
                    "learnings": learnings,
                    "follow_up_questions": follow_up_questions
                }

        except Exception as e:
            print(f"Error processing result: {str(e)}")
            return {
                "learnings": [f"Information about {query}"],
                "follow_up_questions": [f"What are the key aspects of {query}?"]
            }

    async def _are_queries_similar(self, query1: str, query2: str) -> bool:
        """Check if two queries are semantically similar"""
        # Simple string comparison for exact matches
        if query1.lower() == query2.lower():
            return True

        # For very short queries, use substring check
        if len(query1) < 10 or len(query2) < 10:
            return query1.lower() in query2.lower() or query2.lower() in query1.lower()

        # For more complex queries, use Gemini to check similarity
        class SimilarityResult(BaseModel):
            are_similar: bool

        user_prompt = f"""
        Compare these two search queries and determine if they are semantically similar 
        (would likely return similar search results):
        
        Query 1: {query1}
        Query 2: {query2}
        
        Return a JSON object with a single boolean field "are_similar" indicating if the queries are similar.
        """

        generation_config = {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
            "response_mime_type": "application/json",
            "response_schema": SimilarityResult,
        }

        try:
            response = await self.client.aio.models.generate_content(
                model="gemini-2.0-flash",
                contents=user_prompt,
                config=generation_config
            )

            # Get the parsed response using the Pydantic model
            parsed_response = response.parsed
            return parsed_response.are_similar
        except Exception as e:
            print(f"Error comparing queries: {str(e)}")
            # In case of error, assume queries are different to avoid missing potentially unique results
            return False

    async def deep_research(self, query: str, breadth: int, depth: int, learnings: list[str] = [], visited_urls: dict[int, dict] = {}, parent_query: str = None):
        progress = ResearchProgress(depth, breadth)

        # Start the root query
        await progress.start_query(query, depth, parent_query)

        # Adjust number of queries based on mode
        max_queries = {
            "fast": 5,
            "balanced": 10,
            "comprehensive": 7  # Changed from 15 to 7
        }[self.mode]

        queries = await self.generate_queries(
            query,
            min(breadth, max_queries),
            learnings,
            previous_queries=self.query_history
        )

        self.query_history.update(queries)
        unique_queries = list(queries)[:breadth]

        async def process_query(query_str: str, current_depth: int, parent: str = None):
            try:
                # Start this query as a sub-query of the parent
                await progress.start_query(query_str, current_depth, parent)

                result = await self.search(query_str)
                # The search method returns a tuple (formatted_text, sources)
                formatted_text, new_urls = result

                # Add sources to the progress tracker for this query
                if new_urls:
                    sources_list = [
                        {"url": url_data["link"], "title": url_data["title"]}
                        for url_data in new_urls.values()
                        if "link" in url_data and "title" in url_data
                    ]
                    await progress.add_sources(query_str, current_depth, sources_list)

                processed_result = await self.process_result(
                    query=query_str,
                    result=formatted_text,
                    num_learnings=min(5, math.ceil(breadth / 1.5)),
                    num_follow_up_questions=min(5, math.ceil(breadth / 1.5))
                )

                # Record learnings
                for learning in processed_result["learnings"]:
                    await progress.add_learning(query_str, current_depth, learning)

                new_urls = result[1]
                max_idx = max(visited_urls.keys()) if visited_urls else -1
                all_urls = {
                    **visited_urls,
                    **{(i + max_idx + 1): url_data for i, url_data in new_urls.items()}
                }

                # Only go deeper if in comprehensive mode and depth > 1
                if self.mode == "comprehensive" and current_depth > 1:
                    # Reduced breadth for deeper levels, but increased from previous implementation
                    # Less aggressive reduction
                    new_breadth = min(5, math.ceil(breadth / 1.5))
                    new_depth = current_depth - 1

                    # Select most important follow-up questions instead of just one
                    if processed_result['follow_up_questions']:
                        # Take up to 3 most relevant questions instead of just 1
                        follow_up_questions = processed_result['follow_up_questions'][:3]

                        # Process each sub-query
                        for next_query in follow_up_questions:
                            sub_results = await process_query(
                                next_query,
                                new_depth,
                                query_str  # Pass current query as parent
                            )

                            # Merge the sub-results with the current results
                            if sub_results:
                                # Add sub-query learnings to all_urls
                                if "visited_urls" in sub_results:
                                    for url_key, url_data in sub_results["visited_urls"].items():
                                        if url_data['link'] not in [u['link'] for u in all_urls.values()]:
                                            max_idx = max(
                                                all_urls.keys()) if all_urls else -1
                                            all_urls[max_idx + 1] = url_data

                await progress.complete_query(query_str, current_depth)
                return {
                    "learnings": processed_result["learnings"],
                    "visited_urls": all_urls
                }

            except Exception as e:
                print(f"Error processing query {query_str}: {str(e)}")
                await progress.complete_query(query_str, current_depth)
                return {
                    "learnings": [],
                    "visited_urls": {}
                }

        # Process queries concurrently
        tasks = [process_query(q, depth, query) for q in unique_queries]
        results = await asyncio.gather(*tasks)

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

        # Complete the root query after all sub-queries are done
        await progress.complete_query(query, depth)

        # Build the research tree
        research_tree = progress._build_research_tree()

        print(f"Research tree built with {len(all_learnings)} learnings")
        # save the research tree to a file
        with open("research_tree.json", "w") as f:
            json.dump(research_tree, f)

        return {
            "learnings": all_learnings,
            "visited_urls": all_urls,
            "tree": research_tree  # Return the tree structure
        }

    async def generate_final_report(self, query: str, learnings: list[str], visited_urls: dict[int, dict]) -> str:
        # Format learnings for the prompt
        learnings_text = "\n".join([
            f"- {learning}" for learning in learnings
        ])

        user_prompt = f"""
		You are a creative storyteller tasked with transforming research into an engaging and distinctive report.

        Research Query: {query}

        Key Discoveries:
        {learnings_text}

        Craft a captivating report that:

        # CREATIVE APPROACH
        1. Opens with an imaginative introduction that draws readers into the topic
        2. Transforms key discoveries into a compelling narrative with your unique voice
        3. Adds fresh perspectives and unexpected connections between ideas
        4. Experiments freely with tone, style, and expression
        5. Concludes with thought-provoking reflections that linger with the reader

        # FORMATTING TOOLS
        - Create evocative, imaginative headings
        - Use markdown formatting (##, ###, **bold**, *italics*) for visual interest
        - Incorporate blockquotes for emphasis or contrast
        - Deploy bullet points or numbered lists where they enhance clarity
        - Insert tables to organize information in visually appealing ways
        - Use horizontal rules (---) to create dramatic pauses or section breaks

        Feel free to be bold, experimental, and expressive while maintaining clarity and coherence. There are no academic conventions to follow - let your creativity flow!
        """

        generation_config = {
            "temperature": 0.9,  # Increased for more creativity
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }

        try:
            response = await self.client.aio.models.generate_content(
                model="gemini-2.0-flash",
                contents=user_prompt,
                config=generation_config
            )
            return response.text
        except Exception as e:
            print(f"Error generating final report: {str(e)}")
            return f"Error generating report: {str(e)}"

