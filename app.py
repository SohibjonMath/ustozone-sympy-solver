import os, re, hashlib
from typing import Any, Dict, List
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
import sympy as sp
from sympy.calculus.util import continuous_domain
from sympy.solvers.inequalities import solve_univariate_inequality

APP_VERSION = "ustozone-sympy-solver-v1"
app = FastAPI(title="UstozOne SymPy Solver", version=APP_VERSION)

SAFE_FUNCS = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "sqrt": sp.sqrt, "exp": sp.exp, "ln": sp.log, "log": sp.log,
    "Abs": sp.Abs, "abs": sp.Abs, "factorial": sp.factorial,
    "binomial": sp.binomial, "pi": sp.pi, "E": sp.E, "e": sp.E,
    "oo": sp.oo,
}
SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_+\-*/^().,=<>!\[\] {}]+$")
IDENT_RE = re.compile(r"[A-Za-z_]\w*")

class Given(BaseModel):
    name: str
    value: str

class MathSpec(BaseModel):
    solver: str = "sympy"
    operation: str
    expression: str = ""
    lhs: str = ""
    rhs: str = ""
    equations: List[str] = Field(default_factory=list)
    variables: List[str] = Field(default_factory=lambda: ["x"])
    domain: str = "real"
    givens: List[Given] = Field(default_factory=list)
    target: str = ""
    point: str = ""
    lower: str = ""
    upper: str = ""
    order: int = 1
    constraints: List[str] = Field(default_factory=list)
    expectedKind: str = "scalar"
    allowEmpty: bool = False
    unit: str = ""
    trigUnit: str = "radian"
    exact: bool = True
    notes: str = ""

class SolveItem(BaseModel):
    slot: int
    spec: MathSpec

class BatchRequest(BaseModel):
    items: List[SolveItem]


def auth(shared_secret: str | None):
    expected = os.getenv("SOLVER_SHARED_SECRET", "").strip()
    if expected and (shared_secret or "") != expected:
        raise HTTPException(status_code=401, detail="solver secret invalid")


def domain_for(name: str):
    return {
        "real": sp.S.Reals,
        "integer": sp.S.Integers,
        "natural": sp.S.Naturals,
        "positive_real": sp.Interval.open(0, sp.oo),
        "complex": sp.S.Complexes,
    }.get((name or "real").lower(), sp.S.Reals)


def safe_symbols(spec: MathSpec):
    names = set(spec.variables or []) | {g.name for g in spec.givens}
    names |= {"n", "a1", "d", "q", "an", "Sn"}
    out = {n: sp.Symbol(n, real=(spec.domain != "complex")) for n in names if re.fullmatch(r"[A-Za-z_]\w*", n or "")}
    return out


def preprocess(s: str, trig_unit: str):
    s = (s or "").strip().replace("^", "**").replace("−", "-")
    if trig_unit == "degree":
        # Degree trig is made explicit before parsing.
        s = re.sub(r"\bsin\s*\(([^()]*)\)", r"sin(pi*(\1)/180)", s)
        s = re.sub(r"\bcos\s*\(([^()]*)\)", r"cos(pi*(\1)/180)", s)
        s = re.sub(r"\btan\s*\(([^()]*)\)", r"tan(pi*(\1)/180)", s)
    return s


def parse_expr(raw: str, spec: MathSpec, symbols: Dict[str, sp.Symbol]):
    s = preprocess(raw, spec.trigUnit)
    if not s:
        raise ValueError("empty expression")
    if len(s) > 1600 or not SAFE_TOKEN_RE.fullmatch(s):
        raise ValueError("unsafe expression characters")
    allowed = dict(SAFE_FUNCS)
    allowed.update(symbols)
    for ident in IDENT_RE.findall(s):
        if ident not in allowed:
            # Unknown identifiers are symbols only if simple and explicitly referenced.
            if ident in set(spec.variables) | {g.name for g in spec.givens} | {"n", "a1", "d", "q", "an", "Sn"}:
                allowed[ident] = symbols.setdefault(ident, sp.Symbol(ident, real=(spec.domain != "complex")))
            else:
                raise ValueError(f"unknown identifier: {ident}")
    return sp.sympify(s, locals=allowed, evaluate=True)


def substitutions(spec: MathSpec, symbols):
    out = {}
    for g in spec.givens:
        if not re.fullmatch(r"[A-Za-z_]\w*", g.name):
            raise ValueError("invalid given name")
        sym = symbols.setdefault(g.name, sp.Symbol(g.name, real=(spec.domain != "complex")))
        out[sym] = parse_expr(g.value, spec, symbols)
    return out


def equation_from_text(raw: str, spec: MathSpec, symbols, subs):
    text = (raw or "").strip()
    if "=" not in text:
        return sp.Eq(parse_expr(text, spec, symbols).subs(subs), 0)
    left, right = text.split("=", 1)
    return sp.Eq(parse_expr(left, spec, symbols).subs(subs), parse_expr(right, spec, symbols).subs(subs))


def expand_sequence_tokens(expr: str, seq_type: str):
    if seq_type == "arithmetic":
        return re.sub(r"\bSn\b", "(n*(2*a1+(n-1)*d)/2)", re.sub(r"\ban\b", "(a1+(n-1)*d)", expr))
    if seq_type == "geometric":
        # q=1 handled by solver after substitution; expression uses Piecewise-like branch only for Sn direct targets.
        return re.sub(r"\bSn\b", "(a1*(q**n-1)/(q-1))", re.sub(r"\ban\b", "(a1*q**(n-1))", expr))
    return expr


def sorted_finite(values):
    try:
        return sorted(values, key=sp.default_sort_key)
    except Exception:
        return list(values)


def answer_scalar(value, target: str, unit: str):
    value = sp.simplify(value)
    plain = sp.sstr(value).replace("**", "^")
    latex = sp.latex(value)
    prefix = f"{target}=" if target else ""
    up = f" {unit}" if unit else ""
    ul = f"\\,\\text{{{unit}}}" if unit else ""
    return f"{prefix}{plain}{up}", f"{prefix}{latex}{ul}", {"kind": "scalar", "value": sp.srepr(value)}


def answer_set(values, variable: str, unit: str):
    vals = sorted_finite(values)
    if not vals:
        return "Yechim yo‘q.", "\\varnothing", {"kind": "set", "values": []}
    plain_vals = [sp.sstr(sp.simplify(v)).replace("**", "^") for v in vals]
    latex_vals = [sp.latex(sp.simplify(v)) for v in vals]
    if len(vals) == 1:
        return f"{variable}={plain_vals[0]}", f"{variable}={latex_vals[0]}", {"kind":"set","values":[sp.srepr(vals[0])]}
    return f"{variable}=" + ", ".join(plain_vals), f"{variable}\\in\\left\\{{" + ",\\;".join(latex_vals) + "\\right\\}", {"kind":"set","values":[sp.srepr(v) for v in vals]}


def check_constraints(solution: Dict[sp.Symbol, Any], constraints: List[str], spec: MathSpec, symbols):
    if not constraints:
        return True
    subs = dict(solution)
    for raw in constraints:
        s = preprocess(raw, spec.trigUnit)
        m = re.match(r"^(.+?)(!=|<=|>=|=|<|>)(.+)$", s)
        if not m:
            continue
        a = parse_expr(m.group(1), spec, symbols).subs(subs)
        b = parse_expr(m.group(3), spec, symbols).subs(subs)
        op = m.group(2)
        rel = {"!=":sp.Ne,"=":sp.Eq,"<":sp.Lt,">":sp.Gt,"<=":sp.Le,">=":sp.Ge}[op](a,b)
        if rel is sp.S.false or rel == False:
            return False
    return True


def solve_item(item: SolveItem):
    spec = item.spec
    if spec.solver != "sympy":
        return {"slot": item.slot, "status": "unsupported", "valid": True, "reason": "semantic solver requested", "method": "semantic"}
    try:
        symbols = safe_symbols(spec)
        subs = substitutions(spec, symbols)
        op = spec.operation
        target = (spec.target or "").strip()
        dom = domain_for(spec.domain)
        result = None
        method = op
        count = None

        if op == "solve_equation":
            var_name = spec.variables[0] if spec.variables else "x"
            var = symbols.setdefault(var_name, sp.Symbol(var_name, real=(spec.domain != "complex")))
            eq = sp.Eq(parse_expr(spec.lhs, spec, symbols).subs(subs), parse_expr(spec.rhs, spec, symbols).subs(subs)) if (spec.lhs or spec.rhs) else equation_from_text(spec.equations[0], spec, symbols, subs)
            solset = sp.solveset(eq, var, domain=dom)
            if isinstance(solset, sp.FiniteSet):
                vals = [v for v in solset if check_constraints({var:v}, spec.constraints, spec, symbols)]
                count = len(vals)
                if not vals and not spec.allowEmpty:
                    return {"slot":item.slot,"status":"invalid_question","valid":False,"reason":f"{spec.domain} sohada yechim yo‘q.","method":method,"solutionCount":0}
                if spec.expectedKind == "scalar" and len(vals) != 1:
                    return {"slot":item.slot,"status":"invalid_question","valid":False,"reason":f"Yagona javob kutilgan, lekin {len(vals)} ta yechim bor.","method":method,"solutionCount":len(vals)}
                answer, answer_latex, canonical = answer_set(vals, var_name, spec.unit)
            else:
                # Infinite/conditional sets are valid only when set answer expected.
                if spec.expectedKind == "scalar":
                    return {"slot":item.slot,"status":"invalid_question","valid":False,"reason":"Yagona skalyar javob o‘rniga cheksiz/parametrik yechimlar to‘plami chiqdi.","method":method}
                answer = f"{var_name}∈{sp.sstr(solset)}"
                answer_latex = f"{var_name}\\in {sp.latex(solset)}"
                canonical = {"kind":"set","set":sp.srepr(solset)}

        elif op == "solve_system":
            vars_ = [symbols.setdefault(n, sp.Symbol(n, real=(spec.domain != "complex"))) for n in spec.variables]
            eqs = [equation_from_text(e, spec, symbols, subs) for e in spec.equations]
            sols = sp.solve(eqs, vars_, dict=True)
            sols = [s for s in sols if check_constraints(s, spec.constraints, spec, symbols)]
            if not sols and not spec.allowEmpty:
                return {"slot":item.slot,"status":"invalid_question","valid":False,"reason":"Berilgan sohada sistema yechimga ega emas.","method":method,"solutionCount":0}
            if spec.expectedKind == "scalar" and len(sols) != 1:
                return {"slot":item.slot,"status":"invalid_question","valid":False,"reason":f"Yagona javob kutilgan, lekin {len(sols)} ta yechim bor.","method":method,"solutionCount":len(sols)}
            count = len(sols)
            rows=[]; lrows=[]
            for s in sols:
                rows.append("; ".join(f"{v}={sp.sstr(sp.simplify(s[v])).replace('**','^')}" for v in vars_ if v in s))
                lrows.append("\\; ,\\; ".join(f"{sp.latex(v)}={sp.latex(sp.simplify(s[v]))}" for v in vars_ if v in s))
            answer = " yoki ".join(rows) if rows else "Yechim yo‘q."
            answer_latex = "\\text{ yoki }".join(lrows) if lrows else "\\varnothing"
            canonical = {"kind":"system","solutions":[{str(k):sp.srepr(v) for k,v in s.items()} for s in sols]}

        elif op in {"simplify", "evaluate"}:
            expr = parse_expr(spec.expression, spec, symbols).subs(subs)
            result = sp.cancel(sp.factor(sp.simplify(expr))) if op == "simplify" else sp.simplify(expr)
            answer, answer_latex, canonical = answer_scalar(result, target, spec.unit)

        elif op == "limit":
            var_name = spec.variables[0] if spec.variables else "x"
            var = symbols.setdefault(var_name, sp.Symbol(var_name, real=True))
            expr = parse_expr(spec.expression, spec, symbols).subs(subs)
            pt = parse_expr(spec.point or "0", spec, symbols).subs(subs)
            result = sp.simplify(sp.limit(expr, var, pt))
            if result in (sp.oo, -sp.oo, sp.zoo, sp.nan) and spec.expectedKind == "scalar":
                return {"slot":item.slot,"status":"invalid_question","valid":False,"reason":"Chekli skalyar limit kutilgan, lekin limit chekli emas.","method":method}
            answer, answer_latex, canonical = answer_scalar(result, target, spec.unit)

        elif op == "derivative":
            var_name = spec.variables[0] if spec.variables else "x"
            var = symbols.setdefault(var_name, sp.Symbol(var_name, real=True))
            expr = parse_expr(spec.expression, spec, symbols).subs(subs)
            result = sp.diff(expr, var, max(1, int(spec.order or 1)))
            if spec.point:
                result = sp.simplify(result.subs(var, parse_expr(spec.point, spec, symbols).subs(subs)))
            else:
                result = sp.simplify(result)
            answer, answer_latex, canonical = answer_scalar(result, target, spec.unit)

        elif op == "integral":
            var_name = spec.variables[0] if spec.variables else "x"
            var = symbols.setdefault(var_name, sp.Symbol(var_name, real=True))
            expr = parse_expr(spec.expression, spec, symbols).subs(subs)
            if spec.lower or spec.upper:
                lo = parse_expr(spec.lower, spec, symbols).subs(subs)
                hi = parse_expr(spec.upper, spec, symbols).subs(subs)
                result = sp.simplify(sp.integrate(expr, (var, lo, hi)))
            else:
                result = sp.integrate(expr, var)
            answer, answer_latex, canonical = answer_scalar(result, target, spec.unit)

        elif op == "inequality":
            var_name = spec.variables[0] if spec.variables else "x"
            var = symbols.setdefault(var_name, sp.Symbol(var_name, real=True))
            raw = spec.equations[0] if spec.equations else spec.expression
            m = re.match(r"^(.+?)(<=|>=|<|>)(.+)$", preprocess(raw, spec.trigUnit))
            if not m:
                raise ValueError("inequality relation missing")
            left = parse_expr(m.group(1), spec, symbols).subs(subs); right = parse_expr(m.group(3), spec, symbols).subs(subs)
            rel = {"<":sp.Lt,">":sp.Gt,"<=":sp.Le,">=":sp.Ge}[m.group(2)](left,right)
            solset = solve_univariate_inequality(rel, var, relational=False, domain=dom)
            answer = f"{var_name}∈{sp.sstr(solset)}"; answer_latex = f"{var_name}\\in {sp.latex(solset)}"; canonical={"kind":"set","set":sp.srepr(solset)}

        elif op == "domain":
            var_name = spec.variables[0] if spec.variables else "x"
            var = symbols.setdefault(var_name, sp.Symbol(var_name, real=True))
            expr = parse_expr(spec.expression, spec, symbols).subs(subs)
            solset = continuous_domain(expr, var, sp.S.Reals)
            answer = f"D={sp.sstr(solset)}"; answer_latex = f"D={sp.latex(solset)}"; canonical={"kind":"set","set":sp.srepr(solset)}

        elif op in {"arithmetic_progression", "geometric_progression"}:
            seq_type = "arithmetic" if op == "arithmetic_progression" else "geometric"
            var_names = spec.variables or ["n"]
            vars_ = [symbols.setdefault(n, sp.Symbol(n, integer=(spec.domain in {"integer","natural"}), positive=(spec.domain=="natural"))) for n in var_names]
            eqs=[]
            for raw in spec.equations:
                eqs.append(equation_from_text(expand_sequence_tokens(raw, seq_type), spec, symbols, subs))
            sols = sp.solve(eqs, vars_, dict=True)
            filtered=[]
            for s in sols:
                ok=True
                for v in vars_:
                    val=sp.simplify(s.get(v,v))
                    if spec.domain=="natural" and not (val.is_integer is True and val.is_positive is True): ok=False
                    if spec.domain=="integer" and val.is_integer is not True: ok=False
                    if spec.domain=="real" and val.is_real is False: ok=False
                if ok and check_constraints(s,spec.constraints,spec,symbols): filtered.append(s)
            sols=filtered
            if not sols and not spec.allowEmpty:
                return {"slot":item.slot,"status":"invalid_question","valid":False,"reason":f"{spec.domain} sohada talab qilingan progressiya yechimi mavjud emas.","method":method,"solutionCount":0}
            if spec.expectedKind=="scalar" and len(sols)!=1:
                return {"slot":item.slot,"status":"invalid_question","valid":False,"reason":f"Yagona javob kutilgan, lekin {len(sols)} ta yaroqli holat bor.","method":method,"solutionCount":len(sols)}
            target_name=target or (var_names[0] if var_names else "n")
            target_sym=symbols.get(target_name)
            if len(sols)==1 and target_sym in sols[0]:
                answer, answer_latex, canonical=answer_scalar(sols[0][target_sym],target_name,spec.unit)
            else:
                rows=[];lrows=[]
                for s in sols:
                    rows.append("; ".join(f"{k}={sp.sstr(sp.simplify(v)).replace('**','^')}" for k,v in s.items()))
                    lrows.append("\\; ,\\; ".join(f"{sp.latex(k)}={sp.latex(sp.simplify(v))}" for k,v in s.items()))
                answer=" yoki ".join(rows) if rows else "Yechim yo‘q.";answer_latex="\\text{ yoki }".join(lrows) if lrows else "\\varnothing";canonical={"kind":"system","solutions":[{str(k):sp.srepr(v) for k,v in s.items()} for s in sols]}
            count=len(sols)

        elif op == "removable_discontinuity":
            # lhs=numerator, rhs=denominator, variables=[x, parameter], point and optional target value in expression.
            x_name = spec.variables[0] if spec.variables else "x"
            p_name = spec.variables[1] if len(spec.variables)>1 else "a"
            x = symbols.setdefault(x_name, sp.Symbol(x_name, real=True)); p=symbols.setdefault(p_name, sp.Symbol(p_name, real=True))
            num=parse_expr(spec.lhs,spec,symbols).subs(subs); den=parse_expr(spec.rhs,spec,symbols).subs(subs); point=parse_expr(spec.point or "0",spec,symbols).subs(subs)
            candidates=sp.solve([sp.Eq(num.subs(x,point),0),sp.Eq(den.subs(x,point),0)],[p], dict=True)
            target_expr=parse_expr(spec.expression,spec,symbols).subs(subs) if spec.expression else None
            good=[]
            for s in candidates:
                pp=s.get(p)
                if pp is None: continue
                lim=sp.simplify(sp.limit((num/den).subs(p,pp),x,point))
                if target_expr is None or sp.simplify(lim-target_expr.subs(p,pp))==0: good.append(pp)
            if not good and not spec.allowEmpty:
                return {"slot":item.slot,"status":"invalid_question","valid":False,"reason":"Bartaraf etiladigan uzilish va berilgan qo‘shimcha shartni bir vaqtda qanoatlantiruvchi parametr yo‘q.","method":method,"solutionCount":0}
            if spec.expectedKind=="scalar" and len(good)!=1:
                return {"slot":item.slot,"status":"invalid_question","valid":False,"reason":f"Yagona parametr kutilgan, lekin {len(good)} ta holat bor.","method":method,"solutionCount":len(good)}
            answer,answer_latex,canonical=answer_set(good,p_name,spec.unit);count=len(good)

        else:
            return {"slot":item.slot,"status":"unsupported","valid":True,"reason":f"SymPy operation qo‘llab-quvvatlanmaydi: {op}","method":op}

        return {
            "slot": item.slot, "status":"solved", "valid":True, "reason":"", "method":method,
            "answer":answer, "answerLatex":answer_latex, "canonical":canonical, "solutionCount":count,
            "engine":APP_VERSION, "specHash": hashlib.sha256(spec.model_dump_json().encode()).hexdigest()[:20]
        }
    except Exception as exc:
        return {"slot":item.slot,"status":"unsupported","valid":True,"reason":f"SymPy spec yechilmadi: {type(exc).__name__}: {str(exc)[:280]}","method":spec.operation,"engine":APP_VERSION}


@app.get("/health")
def health():
    return {"ok": True, "engine": APP_VERSION, "sympy": sp.__version__}

@app.post("/solve-batch")
def solve_batch(req: BatchRequest, x_solver_secret: str | None = Header(default=None)):
    auth(x_solver_secret)
    if len(req.items) > 60:
        raise HTTPException(status_code=400, detail="max 60 items")
    return {"ok":True,"engine":APP_VERSION,"sympy":sp.__version__,"items":[solve_item(x) for x in req.items]}
