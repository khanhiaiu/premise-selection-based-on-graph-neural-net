import Lean

open Lean Elab Command Term Meta

namespace Lean.Expr
structure GraphState where
  nodes : Array Json := #[]
  map   : Std.HashMap Expr Nat := ∅
abbrev GraphM := StateM GraphState

partial def toGraph (e : Expr) : GraphM Nat := do
  let s ← get
  if let some id := s.map.get? e then return id
  
  let node : Json ← match e with
    | .bvar n            => pure <| Json.mkObj [("kind", "bvar"), ("index", n)]
    | .fvar id           => pure <| Json.mkObj [("kind", "fvar"), ("id", id.name.toString)]
    | .mvar id           => pure <| Json.mkObj [("kind", "mvar"), ("id", id.name.toString)]
    | .sort l            => pure <| Json.mkObj [("kind", "sort"), ("level", toString (repr l))]
    | .const n ls        => 
        let levels := Json.arr (ls.map (fun l => Json.str (toString (repr l)))).toArray
        pure <| Json.mkObj [("kind", "const"), ("name", n.toString), ("levels", levels)]
    | .app f a           => do
        let fId ← toGraph f
        let aId ← toGraph a
        pure <| Json.mkObj [("kind", "app"), ("fn", fId), ("arg", aId)]
    | .lam n t b bi       => do
        let tId ← toGraph t
        let bId ← toGraph b
        pure <| Json.mkObj [("kind", "lam"), ("name", n.toString), ("type", tId), ("body", bId), ("bi", toString (repr bi))]
    | .forallE n t b bi   => do
        let tId ← toGraph t
        let bId ← toGraph b
        pure <| Json.mkObj [("kind", "forall"), ("name", n.toString), ("type", tId), ("body", bId), ("bi", toString (repr bi))]
    | .letE n t v b _    => do
        let tId ← toGraph t
        let vId ← toGraph v
        let bId ← toGraph b
        pure <| Json.mkObj [("kind", "let"), ("name", n.toString), ("type", tId), ("value", vId), ("body", bId)]
    | .lit l             => match l with
        | .natVal n => pure <| Json.mkObj [("kind", "lit"), ("val", n)]
        | .strVal s => pure <| Json.mkObj [("kind", "lit"), ("val", s)]
    | .mdata _ e         => do return (← toGraph e)
    | .proj n i e        => do
        let eId ← toGraph e
        pure <| Json.mkObj [("kind", "proj"), ("type", n.toString), ("index", i), ("expr", eId)]

  let s ← get
  let newId := s.nodes.size
  modify fun s => { nodes := s.nodes.push node, map := s.map.insert e newId }
  return newId

def exportGraph (e : Expr) : Json :=
  let (_, s) := (toGraph e).run {}
  Json.arr s.nodes
end Lean.Expr

elab "#extract_graph " val:term : command => do
  runTermElabM fun _ => do
    let e ← elabTerm val none
    let g := Lean.Expr.exportGraph e
    IO.println g.compress
