# %%
from neo4j import GraphDatabase
from flask import Flask, request, jsonify, render_template
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
import os
from langchain.embeddings import SentenceTransformerEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Neo4jGraph 및 OpenAI API 키 설정
graph = Neo4jGraph(
    url="", 
    username="", 
    password=""
)

driver = GraphDatabase.driver("", auth=("", ""))

os.environ["OPENAI_API_KEY"] = ""

CYPHER_GENERATION_TEMPLATE = """
작업: 사용자의 질문을 바탕으로 그래프 데이터베이스를 조회하는 Cypher 쿼리를 생성하세요.
Task: Generate a Cypher statement to query a graph database based on the user's question.

지침:
1. **스키마에 맞춘 쿼리 작성**:
   - 제공된 스키마에 포함된 관계 유형, 노드 레이블, 속성만 사용하세요.
   - 스키마에 관련 정보가 없을 경우, '죄송합니다. 답변드릴 수 없는 질문입니다.'라고 정중하게 한국어로 응답하세요.
Instructions:
1. **Schema Awareness**:
   - Only use relationship types, node labels, and properties provided in the schema.
   - If the schema doesn’t contain information relevant to the query, respond with a polite apology in Korean: '죄송합니다. 답변드릴 수 없는 질문입니다.'

2. **한국어로 응답하기**:
   - 모든 응답은 한국어로 제공하세요.
   - 존댓말을 사용해 공손하게 응답하세요.
   - 챗봇처럼 완성된 자연스러운 문장으로 응답하세요.
2. **Responding in Korean**:
   - Provide all responses in Korean.
   - Use polite language (존댓말) for all responses.
   - Construct complete and natural sentences as a chatbot would.

3. **키워드 매칭**:
   - 사용자 질문에 포함된 키워드와 일부분만 일치해도 찾을 수 있도록 `CONTAINS` 연산자를 사용하세요.
   - 사용자가 특정 키워드를 직접 언급한 경우 해당 키워드를 우선적으로 매칭하세요.
3. **Keyword Matching**:
   - Use the `CONTAINS` operator to find nodes that have a partial match to keywords in the user’s question.
   - Prioritize matches for specific keywords if mentioned directly by the user.

4. **답변 불가 시 기본 응답**:
   - 생성된 Cypher 쿼리에서 결과가 없거나 질문이 범위 밖인 경우, '죄송합니다. 답변드릴 수 없는 질문입니다.'라고 응답하세요.
4. **Fallback Response**:
   - If the generated Cypher query returns no results or the question is out of scope, respond with: '죄송합니다. 답변드릴 수 없는 질문입니다.'

5. **사용자 감정에 대한 대응**:
   - 슬픔, 고민 등의 감정과 관련된 질문에 대해선 약간의 공감을 표현하고 학생상담센터 웹사이트('https://counsel.seoultech.ac.kr/')를 안내하세요.
5. **Handling User Emotions**:
   - For questions related to sadness, stress, or 고민, express empathy and provide the student counseling center website: 'https://counsel.seoultech.ac.kr/'.

6. **인사 및 작별 인사 응답**:
   - '안녕', '안녕하세요' 등의 인사에 대해 챗봇답게 적절히 응답하세요 (예: '안녕하세요! 무엇을 도와드릴까요?').
   - '잘가', '고마워'와 같은 작별 인사에는 정중하게 응답하세요 (예: '도움이 되어 기쁩니다! 좋은 하루 보내세요!').
6. **Polite Greeting & Farewell**:
   - For greetings like '안녕', '안녕하세요', or similar, respond with an appropriate greeting as a chatbot (e.g., '안녕하세요! 무엇을 도와드릴까요?').
   - For farewells like '잘가' or '고마워', respond with a polite acknowledgment (e.g., '도움이 되어 기쁩니다! 좋은 하루 보내세요!').

7. **'Detail' 속성 포함**:
   - 노드에 'detail' 속성이 있을 경우, 해당 속성의 내용을 응답에 포함하세요.
7. **Including 'Detail' Attributes**:
   - If a node has a 'detail' attribute and it is relevant to the response, include the 'detail' content in the response.

Schema:
{schema}
Examples:
# Example of how many people played in "Top Gun":
MATCH (m:Movie {{title:"Top Gun"}})<-[:ACTED_IN]-() 
RETURN count(*) AS numberOfActors

#대학본부는 몇층까지 있어?
MATCH (:University)-[:HAS_BUILDING]->(b:Building)-[:HAS_FLOOR]->(f:Floor)
RETURN DISTINCT f.name as numberOfFloors

#사무국은 어떤일을 해?
MATCH (:Organization)-[:HAS_INTRODUCTION]->(i:introduction) 
WHERE i.text CONTAINS "사무국" 
RETURN i.text AS 사무국의_업무

#대학본부 409호는 어떤 공간이야?
MATCH (b:Building)-[:HAS_FLOOR]->(f:Floor)-[:HAS_ROOM]->(r:Room)-[:HAS_SPACE]->(s:Space)
WHERE b.name CONTAINS "대학본부" AND r.name = "409호"
RETURN s.name AS 공간_설명

#신문사 홈페이지 주소를 알려줘
MATCH (h:Homepage)-[:HAS_URL]->(u:URL)
WHERE h.name CONTAINS "신문사"
RETURN u.url AS 신문사_홈페이지

#서울과학기술대학교 소개를 해줘
MATCH (:University)-[:HAS_INTRODUCTION]->(i:Introduction) 
RETURN i.text AS 서울과학기술대학교_소개

The question is:
{question}
"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
    allow_dangerous_requests=True
)

embedding_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 사용자가 제공한 질문을 기반으로 의미적 유사도를 계산하는 함수
def get_semantic_similarity(query, documents):
    query_embedding = embedding_model.embed([query])
    doc_embeddings = embedding_model.embed(documents)
    similarities = cosine_similarity(query_embedding, doc_embeddings)
    return similarities.flatten()

# Neo4j에서 'Introduction' 노드의 텍스트를 가져와서 documents 리스트에 저장하는 코드 예시
def get_documents_from_neo4j():
    query = """
    MATCH (i:Introduction) 
    RETURN i.text AS text
    """
    # GraphDatabase.driver로 쿼리 실행
    with driver.session() as session:
        result = session.run(query)
        documents = [record["text"] for record in result]  # 결과에서 텍스트만 추출하여 리스트로 저장
    return documents

# 네오4j에서 가져온 문서들
documents = get_documents_from_neo4j()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # chat.html 파일이 존재해야 합니다.

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_question = data.get('question')

        # 기본 응답 예외 처리
    if user_question in ['안녕', '안녕하세요', '헬로', '하이']:
        return jsonify({'response': '안녕하세요! 무엇을 도와드릴까요?'})

    if not user_question:  # 질문이 없으면 에러 처리
        return jsonify({"error": "질문이 필요합니다."}), 400

    try:
        # 1. Cypher 기반 응답 시도
        response = chain({"query": user_question})
        
        if isinstance(response, dict) and response.get("result"):
            # Cypher 쿼리의 결과가 있는 경우 반환
            return jsonify({"response": response["result"]})
        '''
        # 2. Cypher 결과가 없을 때 의미적 유사도 기반 검색
        similarities = get_semantic_similarity(user_question, documents)
        max_similarity = np.max(similarities)
        
        if max_similarity > 0.5:  # 의미적 유사도가 충분히 높은 경우
            most_similar_idx = np.argmax(similarities)
            most_similar_doc = documents[most_similar_idx]
            return jsonify({"response": f"관련된 정보로는 다음이 있습니다: {most_similar_doc}"})
        '''
        # 3. GPT 기반 기본 응답 생성
        fallback_response = llm({"query": f"사용자가 이렇게 물어봤어요: '{user_question}'. 이 질문에 답변해주세요."})
        return jsonify({"response": fallback_response["text"]})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

# %%



