"""
agent.py — First-Aid & Emergency Health FAQ Agent
Production module: state, knowledge base, nodes, routing, and graph assembly.
Import build_agent() into capstone_streamlit.py.
"""

import os
from dotenv import load_dotenv
from typing import TypedDict, List

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ──────────────────────────────────────────────────────────────
# KNOWLEDGE BASE  (10+ documents, one topic each, 100-500 words)
# ──────────────────────────────────────────────────────────────

DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "CPR",
        "text": (
            "CPR (Cardiopulmonary Resuscitation): Check responsiveness by tapping shoulders and "
            "shouting. If unresponsive, call 112 immediately. Position the heel of your hand on "
            "the centre of the chest (lower half of sternum). Deliver 30 chest compressions at "
            "2 inches deep and 100-120 compressions per minute. Allow full chest recoil between "
            "compressions. Give 2 rescue breaths — tilt head, lift chin, pinch nose, seal mouth, "
            "breathe for 1 second watching for chest rise. Repeat 30:2 cycle until emergency "
            "services arrive or an AED is available. Hands-only CPR (compressions without rescue "
            "breaths) is acceptable for untrained bystanders or when uncomfortable with rescue "
            "breaths. Use an AED as soon as one is available — turn it on and follow the voice "
            "prompts. Do not stop CPR except to use the AED or if the person shows clear signs "
            "of life. CPR doubles or triples survival chances when performed immediately."
        ),
    },
    {
        "id": "doc_002",
        "topic": "Choking",
        "text": (
            "Choking — Heimlich Manoeuvre: Ask 'Are you choking?' If the person cannot speak, "
            "cough, or breathe, act immediately. For adults and children over 1 year: give 5 "
            "firm back blows between shoulder blades using the heel of your hand with the person "
            "leaning forward. Follow with 5 abdominal thrusts — stand behind, arms around waist, "
            "fist above navel below ribs, pull sharply inward and upward. Alternate 5 back blows "
            "and 5 abdominal thrusts until the object is dislodged or the person loses "
            "consciousness. For infants under 1 year: never perform abdominal thrusts. Give 5 "
            "back blows face-down over your forearm, then 5 chest thrusts face-up with two "
            "fingers on the centre of the chest. If the person becomes unconscious: lower them "
            "carefully to the floor and begin CPR. During CPR compressions, look into the mouth "
            "before giving breaths — remove any visible object with a finger sweep. Never perform "
            "blind finger sweeps as this can push the obstruction deeper."
        ),
    },
    {
        "id": "doc_003",
        "topic": "Severe Bleeding",
        "text": (
            "Severe Bleeding Control: Apply direct firm pressure immediately using a clean cloth, "
            "gauze, or bandage. Maintain constant pressure — do not lift to check. If cloth "
            "becomes soaked, add more material on top without removing the first layer. Elevate "
            "the injured limb above the level of the heart if no fracture is suspected. For "
            "life-threatening limb bleeding that does not respond to direct pressure, apply a "
            "tourniquet 5-8 cm above the wound. Tighten until bleeding stops. Note the time of "
            "application on the tourniquet or the person's forehead. Tourniquets should only be "
            "removed by medical professionals. Watch for signs of shock: pale, cold, clammy skin; "
            "rapid weak pulse; rapid shallow breathing; confusion or loss of consciousness; "
            "nausea. If shock is suspected: lay the person flat, elevate legs 30 cm if no spinal "
            "or leg injury, keep warm with a blanket, do not give food or water. Call 112 for all "
            "severe bleeding. Do not remove impaled objects — stabilise in place."
        ),
    },
    {
        "id": "doc_004",
        "topic": "Burns",
        "text": (
            "Burns First Aid: Act immediately. For thermal burns, cool the burn under cool "
            "running water for a minimum of 10 minutes — this removes heat from tissue and "
            "reduces damage. Do not use ice, ice water, butter, toothpaste, or any home remedy "
            "as these worsen tissue damage or cause infection. Remove jewellery and clothing near "
            "the burn unless stuck to the skin. Cover with a sterile non-fluffy dressing or "
            "cling film applied lengthwise. Do not wrap tightly. Seek emergency care for: burns "
            "larger than 3 cm or covering more than 1% body surface; burns on face, hands, feet, "
            "genitals, or joints; all full-thickness (3rd degree) burns — appear white, brown, "
            "or black, no pain sensation; chemical burns — flush with large amounts of water for "
            "20 minutes, do not neutralise with other chemicals; electrical burns — always seek "
            "emergency care even if superficial since internal injury is likely; burns in children, "
            "elderly, or pregnant women. For chemical burns to eyes: irrigate with water for "
            "20 minutes holding eyelids open. Call 112 for serious burns."
        ),
    },
    {
        "id": "doc_005",
        "topic": "Heart Attack",
        "text": (
            "Heart Attack (Myocardial Infarction) Signs and First Aid: Classic symptoms include "
            "central chest pressure, tightness, squeezing, or pain — may radiate to left arm, "
            "neck, jaw, or back. Associated symptoms: shortness of breath, sweating, nausea, "
            "vomiting, light-headedness, or sense of doom. Women may present atypically with "
            "fatigue, indigestion, or jaw pain without classic chest pain. Act immediately — "
            "call 112 without delay. Help the person sit or lie in the most comfortable position "
            "(semi-reclined is typical). Give one adult aspirin (325 mg) to chew slowly — only "
            "if the person is not allergic to aspirin, not on anticoagulants, and is conscious. "
            "Do not give aspirin to children. Keep the person still and calm — physical activity "
            "increases cardiac demand. Loosen tight clothing around the neck and chest. If the "
            "person becomes unconscious and stops breathing normally, begin CPR immediately. "
            "Locate and use an AED if available. Do not leave the person alone. Reassure them "
            "calmly. Note the time symptoms began for the paramedics."
        ),
    },
    {
        "id": "doc_006",
        "topic": "Stroke — FAST",
        "text": (
            "Stroke Recognition and First Aid — FAST Test: Face — ask the person to smile. Does "
            "one side droop? Arms — ask them to raise both arms. Does one arm drift downward? "
            "Speech — ask them to repeat a simple phrase. Is speech slurred or strange? Time — "
            "if ANY of these signs are present, call 112 immediately. Time is critical — each "
            "minute of untreated stroke destroys approximately 2 million brain cells. Additional "
            "stroke signs: sudden severe headache with no known cause (thunderclap headache); "
            "sudden vision loss or double vision in one or both eyes; sudden numbness or weakness "
            "on one side of the body; sudden loss of balance or coordination; sudden confusion. "
            "First Aid steps: call 112 immediately and note the exact time symptoms started. Do "
            "not give food or water — swallowing may be impaired. Do not give aspirin for stroke "
            "unless directed by medical staff — some strokes are haemorrhagic and aspirin worsens "
            "them. If unconscious and breathing: place in recovery position. Begin CPR if not "
            "breathing. Stay with the person until emergency services arrive."
        ),
    },
    {
        "id": "doc_007",
        "topic": "Fractures and Sprains",
        "text": (
            "Fractures and Sprains First Aid: For suspected fractures, do not attempt to "
            "straighten or realign the limb. Immobilise the injury in the position found using "
            "padding, bandages, or improvised splints extending beyond the joints above and below "
            "the fracture. Apply wrapped ice (never directly to skin) for 20 minutes to reduce "
            "swelling. Elevate the injured limb if possible. For open fractures where bone is "
            "visible: cover the wound with a sterile dressing, do not push the bone back, and "
            "call 112. For suspected spinal injury (fall from height, diving accident, vehicle "
            "collision): do NOT move the person unless there is immediate life threat such as "
            "fire. Support the head and neck in the position found. Keep the person still until "
            "emergency services arrive. RICE protocol for sprains: Rest — avoid weight bearing; "
            "Ice — wrapped ice 20 minutes every 2 hours for 48 hours; Compression — elastic "
            "bandage firm but not tight; Elevation — raise above heart level. Seek medical "
            "attention if unable to weight-bear, deformity visible, severe swelling, or "
            "numbness/tingling present."
        ),
    },
    {
        "id": "doc_008",
        "topic": "Anaphylaxis",
        "text": (
            "Anaphylaxis (Severe Allergic Reaction) Signs and First Aid: Anaphylaxis is life-"
            "threatening and requires immediate action. Signs: sudden hives or widespread rash; "
            "swelling of face, lips, tongue, or throat; difficulty breathing, wheezing, or "
            "stridor; rapid weak pulse; drop in blood pressure; pale or bluish skin; dizziness "
            "or loss of consciousness; nausea, vomiting, or abdominal pain. Act immediately: "
            "call 112 first. Use EpiPen (epinephrine auto-injector) in the outer mid-thigh — "
            "can be given through clothing. Hold for 3 seconds, massage the site after. Lay the "
            "person flat with legs elevated — do not allow them to stand. If breathing is "
            "difficult, allow them to sit up slightly. A second EpiPen can be given after 5-15 "
            "minutes if no improvement. Begin CPR if the person becomes unresponsive. The person "
            "MUST go to hospital after EpiPen use — anaphylaxis can have a biphasic reaction "
            "returning hours later. Antihistamines such as Benadryl are NOT a substitute for "
            "epinephrine and will not reverse anaphylaxis. Identify and remove the trigger if "
            "possible."
        ),
    },
    {
        "id": "doc_009",
        "topic": "Poisoning and Overdose",
        "text": (
            "Poisoning and Overdose First Aid: Call India Poison Control immediately: "
            "1800-116-117 (national helpline, free). For ingested poisons: do NOT induce "
            "vomiting unless specifically instructed by poison control — vomiting corrosive "
            "substances causes additional internal burns. Do not give milk, water, or home "
            "antidotes without advice. Keep the container or packaging for poison control. For "
            "inhaled poisons (gas, fumes, smoke): move the person to fresh air immediately. Do "
            "not enter a gas-filled space without breathing apparatus. Begin CPR if not "
            "breathing. For skin or eye contact: remove contaminated clothing using gloves. "
            "Flush skin with large amounts of water for 15-20 minutes. Flush eyes with clean "
            "water for 20 minutes holding eyelids open. Seek medical care. Opioid overdose signs: "
            "pinpoint pupils, unconscious or unresponsive, slow or stopped breathing, blue lips "
            "or fingernails. Naloxone (Narcan) reverses opioid overdose — now available without "
            "prescription in India at selected pharmacies. Administer intranasally or as "
            "injection per instructions. Call 112 regardless — naloxone wears off in 30-90 "
            "minutes and the person may relapse."
        ),
    },
    {
        "id": "doc_010",
        "topic": "Recovery Position",
        "text": (
            "Recovery Position: The recovery position is used for an unconscious person who is "
            "breathing and has no suspected spinal injury. It keeps the airway open and prevents "
            "choking on vomit. Steps: Kneel beside the person. Place the arm nearest you at a "
            "right angle to the body, elbow bent, palm facing up. Bring the far arm across the "
            "chest and hold the back of that hand against the near cheek. With your other hand, "
            "pull up the far knee so the foot is flat on the ground. Keeping the hand against the "
            "cheek, pull on the bent knee to roll the person towards you onto their side. Adjust "
            "the upper leg so both the hip and knee are at right angles. Tilt the head back "
            "slightly to keep the airway open. Monitor breathing continuously until help arrives. "
            "If breathing stops at any time, begin CPR immediately. Reposition to the other side "
            "every 30 minutes to prevent pressure injury. Do NOT use the recovery position if "
            "spinal injury is suspected — maintain spinal alignment. Do NOT use for pregnant "
            "women — position on their left side instead."
        ),
    },
    {
        "id": "doc_011",
        "topic": "Diabetic Emergencies",
        "text": (
            "Diabetic Emergencies — Hypoglycaemia and Hyperglycaemia: Hypoglycaemia (Low Blood "
            "Sugar, below 70 mg/dL): Signs include shakiness, sweating, pale skin, rapid "
            "heartbeat, confusion, irritability, headache, blurred vision, and hunger. If "
            "conscious and able to swallow: give 15-20 g of fast-acting carbohydrates — 150 mL "
            "fruit juice, 3-4 glucose tablets, or 5-6 pieces of hard candy. Wait 15 minutes and "
            "recheck. If still symptomatic, repeat. Once recovered, give a longer-acting snack "
            "such as crackers with peanut butter. If unconscious or cannot swallow: call 112 "
            "immediately. Do not give anything by mouth — risk of aspiration. Place in recovery "
            "position. Hyperglycaemia and Diabetic Ketoacidosis (DKA) — High Blood Sugar: Signs "
            "include excessive thirst and urination, fruity or acetone breath, nausea, vomiting, "
            "abdominal pain, and laboured deep breathing (Kussmaul breathing). This is a medical "
            "emergency — call 112. Do not attempt to treat at home. Keep the person calm and "
            "monitor breathing. If unconscious, use recovery position and begin CPR if needed."
        ),
    },
    {
        "id": "doc_012",
        "topic": "Heat Stroke and Heat Exhaustion",
        "text": (
            "Heat Exhaustion and Heat Stroke First Aid: Heat Exhaustion: Caused by excess heat "
            "and dehydration. Signs: heavy sweating, cool pale clammy skin, fast weak pulse, "
            "nausea, muscle cramps, tiredness, dizziness, headache, fainting. First Aid: move "
            "to a cool shaded environment immediately. Remove excess clothing. Apply cool wet "
            "cloths to skin, especially neck, armpits, and groin. Fan the person. If conscious "
            "and not nauseous, give cool water or electrolyte drinks in small sips. Rest lying "
            "down with legs elevated. If not improving within 15 minutes, treat as heat stroke. "
            "Heat Stroke (Medical Emergency — Life Threatening): Body temperature above 40°C. "
            "Signs: hot red dry skin (no sweating in classic heat stroke), confusion, slurred "
            "speech, loss of consciousness, seizures, rapid strong pulse. Call 112 immediately. "
            "Cool rapidly by any available means: immersion in cool water, ice packs to neck, "
            "armpits, and groin, cool wet sheets, fanning. Cooling is the priority treatment. "
            "Do NOT give fluids to a confused or unconscious person. Do not give aspirin or "
            "paracetamol — they will not help. Continue cooling until emergency services arrive."
        ),
    },
]


# ──────────────────────────────────────────────────────────────
# STATE  — define before any node function
# ──────────────────────────────────────────────────────────────

class CapstoneState(TypedDict):
    question:     str
    messages:     List[dict]
    route:        str
    retrieved:    str
    sources:      List[str]
    tool_result:  str
    answer:       str
    faithfulness: float
    eval_retries: int
    search_results: str


# ──────────────────────────────────────────────────────────────
# MODULE-LEVEL OBJECTS  (initialised once by build_agent)
# ──────────────────────────────────────────────────────────────
# These are set by build_agent() so node functions can reference them.
_llm: ChatGroq | None = None
_embedder: SentenceTransformer | None = None
_collection = None


# ──────────────────────────────────────────────────────────────
# NODE FUNCTIONS  — each independently testable
# ──────────────────────────────────────────────────────────────

def memory_node(state: CapstoneState) -> dict:
    """Append user question to message history; apply sliding window."""
    msgs = list(state.get("messages", []))
    msgs.append({"role": "user", "content": state["question"]})
    return {"messages": msgs[-6:]}  # sliding window prevents token overflow


def router_node(state: CapstoneState) -> dict:
    """LLM-based router: decides retrieve / memory_only / tool."""
    prompt = (
        "You are a router for a First-Aid FAQ agent. Choose exactly ONE route:\n\n"
        "  retrieve     — question asks about first-aid procedures, symptoms, or emergency steps\n"
        "  tool         — question requires CURRENT real-time information, live web data, or today's date/time\n"
        "  memory_only  — question is conversational (greeting, follow-up, thanks) that requires no lookup\n\n"
        f"Question: {state['question']}\n\n"
        "Reply with ONLY one word: retrieve, tool, or memory_only"
    )
    raw = _llm.invoke(prompt).content.strip().lower()
    if "memory" in raw:
        route = "memory_only"
    elif "tool" in raw:
        route = "tool"
    else:
        route = "retrieve"
    return {"route": route}


def retrieval_node(state: CapstoneState) -> dict:
    """Embed question, query ChromaDB for top-3 chunks, format context."""
    q_emb = _embedder.encode([state["question"]]).tolist()
    results = _collection.query(query_embeddings=q_emb, n_results=3)
    chunks = results["documents"][0]
    topics = [m["topic"] for m in results["metadatas"][0]]
    context = "\n\n---\n\n".join(
        f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks))
    )
    return {"retrieved": context, "sources": topics}


def skip_retrieval_node(state: CapstoneState) -> dict:
    """Memory-only route: explicitly clear retrieved context to avoid state bleed."""
    return {"retrieved": "", "sources": []}


def tool_node(state: CapstoneState) -> dict:
    """Web search tool — always returns a string, never raises exceptions."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            raw = list(ddgs.text(
                state["question"] + " first aid health advisory",
                max_results=3
            ))
        if raw:
            result = "\n".join(
                f"• {r['title']}: {r['body'][:250]}" for r in raw
            )
        else:
            result = "Web search returned no results."
    except Exception as e:
        result = f"Web search unavailable: {e}"
    return {"tool_result": result, "search_results": result}


def answer_node(state: CapstoneState) -> dict:
    """Build system prompt with grounding rules; call LLM for final answer."""
    ctx_parts = []
    if state.get("retrieved"):
        ctx_parts.append(f"FIRST-AID KNOWLEDGE BASE:\n{state['retrieved']}")
    if state.get("tool_result"):
        ctx_parts.append(f"WEB SEARCH RESULTS:\n{state['tool_result']}")

    retries = state.get("eval_retries", 0)

    if ctx_parts:
        context = "\n\n".join(ctx_parts)
        escalation = (
            "\n\nIMPORTANT: Your previous answer scored below the faithfulness threshold. "
            "You MUST answer using ONLY the information in the context above. "
            "Do not add any details not explicitly stated in the context."
            if retries > 0 else ""
        )
        system_content = (
            "You are a calm, clear First-Aid assistant. "
            "Answer using ONLY the context provided below. "
            "Do not add any information from your general training knowledge. "
            "If the answer is not in the context, say clearly: "
            "'I don't have specific information on that. Please call emergency services on 112.'"
            f"{escalation}\n\n{context}"
        )
    else:
        system_content = (
            "You are a calm, clear First-Aid assistant. "
            "Answer from the conversation history. "
            "If unsure, advise the user to call 112."
        )

    lc_msgs = [SystemMessage(content=system_content)]
    for m in state.get("messages", [])[:-1]:  # exclude the current question (last entry)
        if m["role"] == "user":
            lc_msgs.append(HumanMessage(content=m["content"]))
        else:
            lc_msgs.append(AIMessage(content=m["content"]))
    lc_msgs.append(HumanMessage(content=state["question"]))

    answer = _llm.invoke(lc_msgs).content
    return {"answer": answer}


def eval_node(state: CapstoneState) -> dict:
    """Rate answer faithfulness 0.0-1.0; increment eval_retries counter."""
    context = state.get("retrieved", "")[:400]
    retries = state.get("eval_retries", 0) + 1

    if not context:
        # No KB context to check against — accept the answer
        return {"faithfulness": 1.0, "eval_retries": retries}

    try:
        raw = _llm.invoke(
            f"Rate the faithfulness of the answer to the context on a scale of 0.0 to 1.0. "
            f"Reply with ONLY a single decimal number between 0.0 and 1.0.\n\n"
            f"Context:\n{context}\n\n"
            f"Answer:\n{state.get('answer', '')[:200]}"
        ).content.strip().split()[0]
        score = max(0.0, min(1.0, float(raw)))
    except Exception:
        score = 0.7  # safe fallback

    return {"faithfulness": score, "eval_retries": retries}


def save_node(state: CapstoneState) -> dict:
    """Append the final answer to message history."""
    msgs = list(state.get("messages", []))
    msgs.append({"role": "assistant", "content": state["answer"]})
    return {"messages": msgs}


# ──────────────────────────────────────────────────────────────
# ROUTING FUNCTIONS  — standalone so they are independently testable
# and satisfy LangGraph's add_conditional_edges() API requirement
# ──────────────────────────────────────────────────────────────

def route_decision(state: CapstoneState) -> str:
    """Read state.route and return the next node name."""
    r = state.get("route", "retrieve")
    if r == "tool":
        return "tool"
    if r == "memory_only":
        return "skip"
    return "retrieve"


def eval_decision(state: CapstoneState) -> str:
    """Return 'save' when faithfulness >= 0.7 OR max retries reached; else 'answer'."""
    FAITHFULNESS_THRESHOLD = 0.7
    MAX_EVAL_RETRIES = 2
    if (
        state.get("faithfulness", 1.0) >= FAITHFULNESS_THRESHOLD
        or state.get("eval_retries", 0) >= MAX_EVAL_RETRIES
    ):
        return "save"
    return "answer"


# ──────────────────────────────────────────────────────────────
# GRAPH ASSEMBLY
# ──────────────────────────────────────────────────────────────

def build_agent():
    """
    Initialise LLM, embedder, ChromaDB, and compile the LangGraph agent.
    Returns: (compiled_app, embedder, collection)
    """
    global _llm, _embedder, _collection

    _llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    _embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Build ChromaDB in-memory collection
    client = chromadb.Client()
    try:
        client.delete_collection("capstone_kb")
    except Exception:
        pass
    _collection = client.create_collection("capstone_kb")

    texts = [d["text"] for d in DOCUMENTS]
    _collection.add(
        documents=texts,
        embeddings=_embedder.encode(texts).tolist(),
        ids=[d["id"] for d in DOCUMENTS],
        metadatas=[{"topic": d["topic"]} for d in DOCUMENTS],
    )

    # Graph assembly
    g = StateGraph(CapstoneState)

    for name, fn in [
        ("memory",   memory_node),
        ("router",   router_node),
        ("retrieve", retrieval_node),
        ("skip",     skip_retrieval_node),
        ("tool",     tool_node),
        ("answer",   answer_node),
        ("eval",     eval_node),
        ("save",     save_node),
    ]:
        g.add_node(name, fn)

    g.set_entry_point("memory")
    g.add_edge("memory", "router")

    g.add_conditional_edges(
        "router",
        route_decision,
        {"retrieve": "retrieve", "skip": "skip", "tool": "tool"},
    )

    for n in ["retrieve", "skip", "tool"]:
        g.add_edge(n, "answer")

    g.add_edge("answer", "eval")

    g.add_conditional_edges(
        "eval",
        eval_decision,
        {"answer": "answer", "save": "save"},
    )

    g.add_edge("save", END)  # ← must not be omitted

    app = g.compile(checkpointer=MemorySaver())
    print(f"Graph compiled successfully — {_collection.count()} documents in knowledge base.")
    return app, _embedder, _collection
