"""Conservative, provenance-preserving Q&A episode boundaries; no LLM inference."""
import re


def norm(text):
    return re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()


ACK = re.compile(
    r'(?:(?:yes|yeah|yep|okay|ok|correct|exactly|right|sure|thanks|thank you|'
    r'i can|we can|i can hear you|we can hear you|that is right|that s right|'
    r'that is correct|that s correct|please go ahead|go ahead)\s*)+'
)
NEW_QUESTION = re.compile(r'\b(?:next|another|separate|second|follow up)\s+(?:question|topic)|\bjust as a follow up\b')
QUESTION_CUE = re.compile(r'\b(?:what|when|where|why|how|could|can you|would you|do you|'
                          r'talk about|tell us|give us|help me|help us|wondering|curious|'
                          r'walk us|explain|quantify)\b')
INDIRECT_DETAIL_REQUEST = re.compile(
    r'\b(?:i|we) (?:am|are|was|were|would be) (?:just |also )?'
    r'(?:looking|hoping) for (?:just )?(?:kind of )?'
    r'(?:a little bit |a little |a bit |an? |some )?'
    r'(?:more |additional |further )?'
    r'(?:details?|colou?r|clarification|update|breakdown|explanation) '
    r'(?:on|about|regarding|around|of) \S'
)


def acknowledgement(text):
    return '?' not in text and bool(ACK.fullmatch(norm(text)))


def audio_check(text):
    pieces = [norm(p) for p in re.split(r'[.!?]+', text) if norm(p)]
    cue = re.compile(r'(?:can you hear (?:us|me)(?: (?:okay|ok|now))?|'
                     r'are you (?:there|on the line)|let me (?:do a quick check|check))')
    intro = re.compile(r'(?:yes|hi|hello)(?: [a-z]+){0,2}|(?:this is)(?: [a-z]+){1,3}')
    return bool(pieces) and any(cue.fullmatch(p) for p in pieces) and all(
        cue.fullmatch(p) or intro.fullmatch(p) for p in pieces)


def source_quality_reasons(text):
    """Observable source defects, not a judgment that content is meaningless."""
    reasons = []
    if re.search(r'\([^)]*\b(?:inaudible|indiscernible|multiple speakers|technical difficulties)\b[^)]*\)', text, re.I):
        reasons.append('source_text_gap')
    if re.search(r'(?:\.{3}|\u2026|--|\u2014)\s*[.!?]?\s*$', text):
        reasons.append('trailing_source_fragment')
    return reasons


def clarification_only(text):
    """Recognize complete clarification passages, never a procedural prefix alone."""
    pieces = [norm(p) for p in re.split(r'[.!?]+', text) if norm(p)]
    cue = re.compile(
        r'(?:could|can|would) you (?:please )?(?:repeat|clarify)'
        r'(?: (?:the|your) question| (?:that|it))?(?: again)?(?: please)?|'
        r'(?:could|can|would) you (?:please )?say (?:that|it) again(?: please)?|'
        r'i (?:missed|did not hear|didn t hear|did not catch|didn t catch) '
        r'(?:your|the|that) question|'
        r'(?:i )?i (?:m|am) not sure i (?:understand|follow) (?:your|the) question'
    )
    courtesy = re.compile(r'(?:i m|i am) sorry|sorry|well|please|thanks|thank you')
    prefix = re.compile(r'^(?:well )?(?:(?:i m|i am) sorry |sorry )?')
    matches = [bool(cue.fullmatch(prefix.sub('', p, count=1))) for p in pieces]
    return bool(pieces) and any(matches) and all(
        matched or courtesy.fullmatch(p) for p, matched in zip(pieces, matches)
    )


def procedure_kind(text):
    """Only high-specificity procedural patterns; unknown content stays visible."""
    n = norm(text)
    greeting = re.compile(r'(?:(?:yes|yeah|sure|good|thanks|thank you|we re fine|we re doing good|how are you)\s*|(?:hi|hello|hey|good morning|good afternoon)(?: [a-z]+)?\s*)+')
    if re.search(r'\b(?:hi|hello|hey|good morning|good afternoon)\b', n) and greeting.fullmatch(n):
        return 'greeting_only'
    if audio_check(text):
        return 'audio_check'
    if re.fullmatch(r"give [a-z]+ a second (?:he s|she s|they re) looking (?:that|it) up", n):
        return 'lookup_deferral'
    if re.fullmatch(r'[a-z]+ do you have (?:a good estimate|an estimate|a number|the number)', n):
        return 'delegation_only'
    if re.fullmatch(r'[a-z]+ (?:(?:can|could|would) you (?:please )?|(?:do )?you want to )take (?:it|that|this)', n):
        return 'delegation_only'
    if re.fullmatch(r'i(?: ve got| have) to turn to [a-z]+ for that one', n):
        return 'delegation_only'
    if re.fullmatch(r'i (?:am going to|m going to|will|ll) ask [a-z]+(?: [a-z]+)? to', n) and re.search(r'(?:\.{3}|\u2026)\s*$', text):
        return 'delegation_only'
    if clarification_only(text):
        return 'clarification_request'
    if (len(n.split()) <= 60 and
        re.match(r'(?:so )?if i understand your question\b', n) and
        not re.search(r'[.!?]\s+\S', text) and
        re.search(r'\b(?:do i have that right|is that correct)$', n)):
        return 'clarification_request'
    if re.fullmatch(r'do you want to (?:speak|attend|join)(?: or do you want to (?:speak|attend|join))?(?: it)?', n):
        return 'logistics_question'
    if re.fullmatch(r'(?:give me|what is|what s) the (?:conference )?call number(?: and (?:the )?password)?', n):
        return 'logistics_reply'
    if n in ('well', 'uh', 'um'):
        return 'incomplete_fragment'
    invitation = re.compile(r'(?:now )?(?:we would be happy to take|we will take|we are ready for) (?:any |your )?questions(?: you may have)?')
    pieces = [norm(p) for p in re.split(r'[.!?]+', text) if norm(p)]
    if pieces and any(invitation.fullmatch(p) for p in pieces) and all(
        invitation.fullmatch(p) or re.fullmatch(r'(?:thanks|thank you)(?: [a-z]+){0,2}', p) for p in pieces):
        return 'question_invitation'
    # A closing is only removable when EVERY sentence is procedural. This keeps
    # brief substantive disclosures appended to thanks from being silently lost.
    pieces = [norm(p) for p in re.split(r'[.!?]+', text) if norm(p)]
    closing = re.compile(
        r'(?:well )?thanks for joining us this morning|'
        r'if you have any further questions please don t hesitate to call our investor relations team'
        r'(?: and thanks for your support)?|'
        r'(?:well )?i d like to thank (?:everyone|everybody) for (?:their|your) attention today|'
        r'(?:feel )?please feel free to reach out to our (?:ir|investor relations) team with any questions'
        r'(?: and have a great day)?|'
        r'(?:and )?have a great day(?: and thank you for your time)?|'
        r'thanks for your time today|'
        r'(?:thank you (?:all|everyone|everybody)|thanks everyone) for joining (?:us|the call)|'
        r'(?:thanks|thank you)(?: everyone| all| [a-z]+)?(?: for joining us)?|'
        r'all right(?: [a-z]+)?|'
        r'we (?:certainly )?(?:again )?appreciate (?:everyone s|everybody s|your) (?:interest|support)'
        r'(?: and i m certain we re going to field follow up questions later)?|'
        r'thanks for (?:your )?(?:interest|support)|'
        r'we appreciate you being with us today(?: and your continued interest in (?:[a-z]+ ){0,3}[a-z]+)?|'
        r'we look forward to reporting further advances in [a-z]+ and the opportunity to talk to you again|'
        r'have a (?:great|good|nice) day|we will see you soon|'
        r'we know it s a busy day out there in the market|'
        r'and i m certain we re going to field follow up questions later'
    )
    if pieces and all(closing.fullmatch(p) for p in pieces):
        return 'closing_only'
    return ''


def analyst_question_candidate(text):
    return not procedure_kind(text) and (
        '?' in text or bool(QUESTION_CUE.search(norm(text))) or
        any(INDIRECT_DETAIL_REQUEST.search(norm(sentence))
            for sentence in re.split(r'[.!?]+', text))
    )


def bare_analyst_handoff(operator, following):
    """Accept only a complete name-and-firm announcement matching the next label."""
    label = re.sub(r'\s*\[\d+\]\s*$', '', following.get('text_name', ''))
    identity = re.split(r'\s+-\s+', label, maxsplit=1)[0]
    if ',' not in identity:
        return False
    name, firm = identity.split(',', 1)
    if len(norm(name).split()) < 2 or 'unidentified' in norm(name):
        return False
    # Only optional legal suffixes, not arbitrary prefix/fuzzy matching.
    firm = re.sub(r'\s+(?:and company|company|inc|llc|ltd|corp|corporation)$', '', norm(firm))
    announcement = norm(operator.get('text_contents', ''))
    return bool(firm) and announcement in (f'{norm(name)} {firm}', f'{norm(name)} of {firm}')


def company_line_opening_bridge(row, following):
    """Recognize only fully procedural, corroborated company-line interruptions."""
    label = re.sub(r'\s*\[\d+\]\s*$', '', row.get('text_name', '')).strip()
    if label.casefold() != 'unidentified company representative':
        return False
    text = norm(row.get('text_contents', ''))
    confirmation = norm(following.get('text_contents', ''))
    return bool(re.fullmatch(
        r'(?:he|she) should be on the operator just needs to open (?:his|her) line', text
    ) and re.fullmatch(
        r'operator (?:can|could) you please open [a-z]+(?: [a-z]+){0,3} s line', confirmation
    ))


def bounded_bare_handoff(rows, pos, is_analyst, is_operator):
    """Boundary evidence only: never assign the announced person's identity."""
    text = rows[pos].get('text_contents', '').strip()
    match = re.fullmatch(r"([A-Z][a-zA-Z'-]+(?: [A-Z][a-zA-Z'.-]+){1,3}), ([A-Za-z][A-Za-z &.'-]{1,65})\.?", text)
    if not match or norm(match[1]).split()[0] in ('please', 'thank', 'operator', 'our', 'next'):
        return False
    following = pos + 1
    if following < len(rows) and not is_analyst(rows[following]) and not is_operator(rows[following]):
        if procedure_kind(rows[following].get('text_contents', '')) != 'greeting_only':
            return False
        following += 1
    if following >= len(rows) or not is_analyst(rows[following]):
        return False
    label = re.sub(r'\s*\[\d+\]\s*$', '', rows[following].get('text_name', ''))
    identity = re.split(r'\s+-\s+', label, maxsplit=1)[0]
    if ',' not in identity:
        return False
    firm = norm(identity.split(',', 1)[1])
    announced = norm(match[2])
    # Whole-token institutional prefix permits UBS / UBS Securities LLC,
    # but not UBS / UBSmith. The speaker may explicitly be a substitute.
    return len(announced) >= 3 and (firm == announced or firm.startswith(announced + ' '))


def collect_episode(rows, start, *, is_analyst, is_operator, is_question_handoff, speaker_key, is_unresolved=None):
    """Return one episode plus next cursor; unknown links are never high-tier."""
    analyst = speaker_key(rows[start])
    questions = [rows[start]]
    context, procedural, flags = [], [], set()
    pos = start + 1
    # Consecutive turns from a different analyst are never joined.
    while pos < len(rows) and is_analyst(rows[pos]) and speaker_key(rows[pos]) == analyst:
        questions.append(rows[pos])
        pos += 1
    while pos < len(rows):
        row = rows[pos]
        if is_unresolved and is_unresolved(row):
            following = rows[pos + 1] if pos + 1 < len(rows) else None
            if (context and following is not None and not is_unresolved(following)
                    and not is_analyst(following) and not is_operator(following)
                    and company_line_opening_bridge(row, following)):
                context.append(row)
                procedural.append(row)
                flags.add('UNIDENTIFIED_COMPANY_LOGISTICS_LINK_REVIEW')
                pos += 1
                continue
            flags.add('UNRESOLVED_SPEAKER_BOUNDARY_REVIEW')
            break
        if is_operator(row):
            # Formal closing invitation is not part of the preceding answer.
            if re.search(r'\b(?:no further questions|any (?:additional )?closing remarks|any closure or further remarks|any closing or further remarks|'
                         r'(?:that|this) (?:will )?(?:conclude|concludes) the (?:q\s*&\s*a|question(?:s)? and answer(?:s)?) session)\b',
                         row.get('text_contents', ''), re.I):
                flags.add('OPERATOR_CLOSING_BOUNDARY')
                break
            if is_question_handoff(row):
                break
            if bounded_bare_handoff(rows, pos, is_analyst, is_operator):
                break
            if pos + 1 < len(rows) and is_analyst(rows[pos + 1]) and bare_analyst_handoff(row, rows[pos + 1]):
                break
            flags.add('OPERATOR_INTERRUPTION_REVIEW')
            context.append(row)
            pos += 1
            continue
        if not is_analyst(row):
            context.append(row)
            pos += 1
            continue
        if speaker_key(row) != analyst:
            break
        previous = next((r for r in reversed(context) if not is_analyst(r) and not is_operator(r)), None)
        if previous is None:
            break
        text = row.get('text_contents', '')
        prior_kind = procedure_kind(previous.get('text_contents', ''))
        if any(is_operator(r) for r in context):
            break
        if NEW_QUESTION.search(norm(text)):
            break
        if (re.search(r'(?:--|\u2014)\s*$', previous.get('text_contents', '')) and
            re.fullmatch(r'(?:do you have any|can you|could you)\s*(?:--|\u2014)', text.strip(), re.I) and
            pos + 1 < len(rows) and not is_analyst(rows[pos + 1]) and
            not is_operator(rows[pos + 1]) and speaker_key(rows[pos + 1]) == speaker_key(previous)):
            context.append(row)
            flags.add('INTERRUPTED_CONTINUATION_LINK_REVIEW')
            pos += 1
            continue
        if acknowledgement(text):
            procedural.append(row)
            context.append(row)
            flags.add('ANALYST_ACKNOWLEDGEMENT_RETAINED_IN_CONTEXT')
            pos += 1
            continue
        if prior_kind in ('audio_check', 'clarification_request'):
            # A substantive restatement after a clarification request may refine
            # the target, but merits review rather than an automatic clean label.
            questions.append(row)
            context.append(row)
            flags.add('QUESTION_RESTATEMENT_LINK_REVIEW')
            pos += 1
            continue
        if prior_kind == 'logistics_question' and procedure_kind(text) == 'logistics_reply':
            procedural.append(row)
            context.append(row)
            flags.add('LOGISTICS_INTERRUPTION_LINK_REVIEW')
            pos += 1
            continue
        # Generic CEO counterquestions require contextual review; do not invent
        # a semantic relation or silently discard the preceding question.
        if previous.get('text_contents', '').rstrip().endswith('?') and len(previous.get('text_contents', '').split()) <= 40:
            flags.add('UNRESOLVED_CEO_COUNTERQUESTION_REVIEW')
        break
    return questions, context, procedural, flags, pos
