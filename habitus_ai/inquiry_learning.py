"""Positive, delayed learning from a local information-seeking conversation.

Teacher assessments concern the learner's expressed gap, question, and update.
They are separate from private processing and are never inferred from an exact
phrase, a reference answer, or mere generation. Only actual incoming feedback
can reinforce a previously externalized output's Habitus route.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import struct

from .calibrated_stream import CalibratedStreamBackend, FeedbackSchedule
from .graph import GraphSide, OUTPUT_NODE_IDS, SELF_ID
from .habitus_stream import encode_json, sha, unit
from .recurrent_flow import RecurrentFlow, digest
from .stream_data import StreamDataCollector
from .types import ExperienceReturn, InputTrunk, OutputDecision, OutputTrunk


MAGIC = b"HABINQ1\0"
STAGE_REWARD = {"acknowledged_gap": 0.10, "useful_question": 0.35,
                "coherent_update": 0.55, "closed_without_progress": 0.0}


@dataclass(frozen=True)
class InquiryFeedback:
    emission_id: str
    stage: str
    note: str
    progress: float = 1.0
    followup_emission_id: str | None = None

    def __post_init__(self):
        if self.stage not in STAGE_REWARD or not self.note.strip():
            raise ValueError("feedback needs a recognized stage and an assessment note")
        if not math.isfinite(self.progress) or not 0 <= self.progress <= 1:
            raise ValueError("feedback progress must be finite and between zero and one")


class InquiryStreamBackend(CalibratedStreamBackend):
    """Reinforced contextual output fibers in the same saved Habitus graph."""

    def __init__(self, backend, *, schedule=None, inquiry_gain=0.005):
        if not math.isfinite(inquiry_gain) or not 0 <= inquiry_gain <= 0.05:
            raise ValueError("invalid inquiry feedback gain")
        super().__init__(backend, schedule=schedule or FeedbackSchedule(0.005))
        if self.schedule.peak_gain + inquiry_gain > 0.25:
            raise ValueError("combined stream gain exceeds its bound")
        self.inquiry_gain = float(inquiry_gain)
        self.identity += ":inquiry-v1:" + sha(encode_json({"gain": inquiry_gain, "rewards": STAGE_REWARD}))
        self.inquiry_enabled = True
        self.inquiry_steps = 0
        self.context = None
        self._feedback_cache = None
        self.bridge.mind.store.connection.executescript("""
            CREATE TABLE inquiry_emissions (
                emission_id TEXT PRIMARY KEY, input_number INTEGER NOT NULL,
                public_sha256 TEXT NOT NULL, context_node TEXT NOT NULL,
                motor_node TEXT NOT NULL, motor_edge TEXT NOT NULL,
                baseline_log_strength REAL NOT NULL, cycle_id TEXT NOT NULL UNIQUE,
                output_record_id TEXT NOT NULL, context_json TEXT NOT NULL
            );
            CREATE TABLE inquiry_feedback (
                emission_id TEXT NOT NULL, stage TEXT NOT NULL,
                input_number INTEGER NOT NULL, input_record_id TEXT NOT NULL,
                reward REAL NOT NULL, assessment_json TEXT NOT NULL,
                PRIMARY KEY (emission_id, stage)
            );
        """)

    @property
    def connection(self):
        return self.bridge.mind.store.connection

    def admit_input(self, message):
        super().admit_input(message)
        self.context = None
        self._feedback_cache = None

    def learned_feedback(self):
        if not self.inquiry_enabled or self.context is None:
            return [0.0] * self.base.embedding_dimension
        if self._feedback_cache is not None:
            return self._feedback_cache.copy()
        mind = self.bridge.mind
        readout = [0.0] * self.base.embedding_dimension
        mass = 0.0
        # Fixed order and bounded support make this independent of audit UUIDs.
        for row in self.connection.execute("SELECT * FROM inquiry_emissions ORDER BY emission_id"):
            edge = mind.store.get_edge(row['motor_edge'])
            support = 1.0 - math.exp(-max(0.0, edge.log_strength - row['baseline_log_strength']))
            if support <= 0:
                continue
            context = json.loads(row['context_json'])
            similarity = max(0.0, sum(a * b for a, b in zip(self.context, context))) ** 8
            weight = support * similarity
            motor = mind.store.get_concept(row['motor_node']).embedding
            readout = [left + weight * right for left, right in zip(readout, motor)]
            mass += weight
        self._feedback_cache = [value / max(1.0, mass) for value in readout]
        return self._feedback_cache.copy()

    def consume_generated(self, token, *, private):
        if token != self.eos_token_id and self.context is None:
            self.context = unit(self.base.stream_context())
            self._feedback_cache = None
        learned = self.learned_feedback() if private and token != self.eos_token_id else []
        if not learned or not any(learned) or self.inquiry_gain == 0:
            return super().consume_generated(token, private=private)
        ordinary = self.bridge.feedback()
        gain = self.current_gain()
        total_gain = gain + self.inquiry_gain
        field = [(gain * a + self.inquiry_gain * b) / total_gain for a, b in zip(ordinary, learned)]
        result = self.base.consume_with_feedback(token, field, gain=total_gain)
        if gain and any(ordinary):
            self.bridge.feedback_steps += 1
            self.requested_gain_sum += gain
        self.bridge.observe_private(self.base.stream_embedding([token]))
        self.private_since_input += 1
        self.private_steps += 1
        self.inquiry_steps += 1
        return result

    def register_output(self, emission):
        if emission.get('origin') != 'model_public_output' or not emission['text'].strip():
            raise ValueError("inquiry learning requires a public model emission")
        if emission['input_number'] != self.bridge.inputs or self.context is None:
            raise ValueError("public emission lacks its current numeric context")
        number = self.connection.execute('SELECT COUNT(*) FROM inquiry_emissions').fetchone()[0] + 1
        emission_id = f"inquiry-emission:{number}"
        context_id, motor_id = emission_id + ':context', emission_id + ':motor'
        mind, pulse = self.bridge.mind, self.bridge.mind.pulse
        mind.graph.add_concept(context_id, context_id, embedding=self.context,
                               output_trunks=(OutputTrunk.SPEAK,), kind='inquiry_context', pulse=pulse)
        mind.graph.add_concept(motor_id, motor_id, embedding=unit(self.base.stream_embedding(self.encode(emission['text']))),
                               kind='inquiry_motor', pulse=pulse)
        edge = mind.graph.add_relation(context_id, motor_id, side=GraphSide.OUTPUT, pulse=pulse)
        trace = mind.graph.trace_explicit_path(pulse_id=emission_id, side=GraphSide.OUTPUT,
            path_node_ids=(SELF_ID, OUTPUT_NODE_IDS[OutputTrunk.SPEAK], context_id, motor_id),
            endpoint_score=1.0, mark_active=True)
        public_hash = sha(emission['text'].encode())
        cycle = mind.begin_output_cycle('public-emission-sha256:' + public_hash,
            OutputDecision(emission_id, OutputTrunk.SPEAK, 1.0, trace),
            cycle_id=emission_id + ':cycle', embedding=unit(self.base.stream_embedding(self.encode(emission['text']))),
            credited_edge_ids=(edge.edge_id,), metadata={'origin': 'model_public_output',
                'public_sha256': public_hash, 'local_dialogue_delivery': True,
                'verified_file_or_tool_action': False, 'input_number': emission['input_number']})
        self.connection.execute('INSERT INTO inquiry_emissions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (emission_id, emission['input_number'], public_hash, context_id, motor_id, edge.edge_id,
             edge.log_strength, cycle.cycle_id, cycle.output_record_id, json.dumps(self.context)))
        self.connection.commit()
        self._feedback_cache = None
        return emission_id

    def receive_feedback(self, feedback, input_receipt):
        row = self.connection.execute('SELECT * FROM inquiry_emissions WHERE emission_id=?',
                                      (feedback.emission_id,)).fetchone()
        if row is None or input_receipt['input_number'] <= row['input_number']:
            raise ValueError("feedback must follow an existing public emission")
        mind = self.bridge.mind
        cycle = mind.store.get_experience_cycle(row['cycle_id'])
        if cycle.status != 'open':
            raise ValueError("inquiry output is already closed")
        if self.connection.execute('SELECT 1 FROM inquiry_feedback WHERE emission_id=? AND stage=?',
                                   (feedback.emission_id, feedback.stage)).fetchone():
            raise ValueError("this inquiry stage already received credit")
        records = [record for record in mind.store.list_records()
                   if record.metadata.get('message_sha256') == input_receipt['message_sha256']
                   and record.metadata.get('origin') == 'external_language']
        if not records or input_receipt['input_number'] != self.bridge.inputs:
            raise ValueError("feedback requires an admitted current external input")
        record = records[-1]
        prior = self.connection.execute("SELECT input_number FROM inquiry_feedback WHERE emission_id=? AND stage='useful_question' AND reward>0",
                                        (feedback.emission_id,)).fetchone()
        if feedback.stage == 'coherent_update':
            followup = self.connection.execute('SELECT * FROM inquiry_emissions WHERE emission_id=?',
                                               (feedback.followup_emission_id,)).fetchone()
            if prior is None or followup is None or not prior[0] <= followup['input_number'] < input_receipt['input_number']:
                raise ValueError("coherent update needs an information return and a later public update")
        reward = STAGE_REWARD[feedback.stage] * feedback.progress
        terminal = feedback.stage in {'coherent_update', 'closed_without_progress'}
        metadata = {'origin': 'local_teacher_assessment', 'stage': feedback.stage,
                    'assessment': asdict(feedback), 'verified_scope': 'external feedback was received in the local teaching dialogue',
                    'subjective_report_is_objective_truth': False, 'private_text_was_replayed': False}
        mind.store.add_experience_return(ExperienceReturn(row['cycle_id'], record.record_id,
            InputTrunk.HEAR, feedback.stage, reward, True, terminal, mind.pulse, metadata))
        mind.store.add_record_link(record.record_id, 'responds_to_inquiry', row['output_record_id'], metadata)
        outcome = mind.record_outcome(OutputDecision(row['cycle_id'], OutputTrunk.SPEAK, 1.0, None),
            stability_delta=reward, verified=True, proposal_id=row['output_record_id'],
            receipt_id=record.record_id, credited_edge_ids=(row['motor_edge'],), metadata=metadata)
        self.connection.execute('INSERT INTO inquiry_feedback VALUES (?, ?, ?, ?, ?, ?)',
            (feedback.emission_id, feedback.stage, input_receipt['input_number'], record.record_id,
             reward, json.dumps(asdict(feedback), sort_keys=True)))
        self.connection.commit()
        self._feedback_cache = None
        return {'emission_id': feedback.emission_id, 'stage': feedback.stage, 'reward': reward,
                'input_record_id': record.record_id, 'outcome_id': outcome.outcome_id,
                'motor_edge': row['motor_edge'], 'motor_log_strength': mind.store.get_edge(row['motor_edge']).log_strength}

    def inquiry_summary(self):
        emissions = [dict(row) for row in self.connection.execute('SELECT * FROM inquiry_emissions ORDER BY emission_id')]
        feedback = [dict(row) for row in self.connection.execute('SELECT * FROM inquiry_feedback ORDER BY emission_id, stage')]
        computational = {'emissions': [{key: value for key, value in row.items() if key != 'output_record_id'} for row in emissions],
                         'feedback': [{key: value for key, value in row.items() if key != 'input_record_id'} for row in feedback],
                         'context': self.context, 'enabled': self.inquiry_enabled, 'steps': self.inquiry_steps}
        return {'public_outputs': len(emissions), 'feedback_events': len(feedback),
                'reward_total': sum(row['reward'] for row in feedback),
                'reinforced_motors': sum(self.bridge.mind.store.get_edge(row['motor_edge']).log_strength > row['baseline_log_strength'] for row in emissions),
                'inquiry_steps': self.inquiry_steps, 'enabled': self.inquiry_enabled,
                'feedback_norm': math.sqrt(sum(value * value for value in self.learned_feedback())),
                'computational_sha256': sha(encode_json(computational))}

    def dump_state(self):
        parent = super().dump_state()
        meta = encode_json({'identity': self.identity, 'context': self.context,
                            'inquiry_enabled': self.inquiry_enabled, 'inquiry_steps': self.inquiry_steps,
                            'parent_sha256': sha(parent)})
        return MAGIC + struct.pack('>Q', len(meta)) + meta + parent

    def load_state(self, data):
        offset = len(MAGIC) + 8
        if not data.startswith(MAGIC) or len(data) <= offset:
            raise ValueError("invalid inquiry checkpoint header")
        length = struct.unpack('>Q', data[len(MAGIC):offset])[0]
        if not 0 < length < len(data) - offset:
            raise ValueError("invalid inquiry checkpoint length")
        meta = json.loads(data[offset:offset + length])
        context = meta['context']
        if (meta['identity'] != self.identity or not isinstance(meta['inquiry_enabled'], bool)
                or not isinstance(meta['inquiry_steps'], int) or meta['inquiry_steps'] < 0
                or (context is not None and (len(context) != self.base.embedding_dimension
                    or not all(math.isfinite(value) for value in context)))):
            raise ValueError("inquiry checkpoint identity or state mismatch")
        parent = data[offset + length:]
        if sha(parent) != meta['parent_sha256']:
            raise ValueError("inquiry checkpoint checksum mismatch")
        super().load_state(parent)
        self.context, self.inquiry_enabled, self.inquiry_steps = context, meta['inquiry_enabled'], meta['inquiry_steps']
        self._feedback_cache = None


class InquiryFlow(RecurrentFlow):
    """Capture actual public output and attach subsequently received teaching."""

    def pulse(self, *, maximum_tokens=None):
        before = self.checkpoint()
        try:
            receipt = super().pulse(maximum_tokens=maximum_tokens)
            for emission in receipt['public_emissions']:
                emission['inquiry_emission_id'] = self.backend.register_output(emission)
            receipt['state_after_sha256'] = digest(self.backend.dump_state())
            return receipt
        except BaseException:
            self.restore(before)
            raise

    def hear(self, message, *, feedback=()):
        before = self.checkpoint()
        try:
            receipt = super().hear(message)
            receipt['inquiry_feedback'] = [self.backend.receive_feedback(item, receipt) for item in feedback]
            receipt['state_after_sha256'] = digest(self.backend.dump_state())
            return receipt
        except BaseException:
            self.restore(before)
            raise


class InquiryDataCollector(StreamDataCollector):
    def __init__(self):
        super().__init__()
        self.rows['learned_inquiry_feedback'] = []

    def capture(self, backend, **kwargs):
        import numpy as np
        super().capture(backend, **kwargs)
        self.rows['learned_inquiry_feedback'].append(np.asarray(backend.learned_feedback(), dtype=np.float32))
        self.frames[-1]['inquiry'] = backend.inquiry_summary()
