"""
app.py
-----------------------------------------
Flask backend.

Responsibilities:
  1. Serve the game page (templates/index.html + static files).
  2. Run the Genetic Algorithm (ai.py) one generation at a time
     whenever the frontend asks for it, and return the best
     weights found so far.
  3. The frontend (script.js) then uses those weights to control
     the bird in the LIVE, visual game running in the browser.

This is intentionally simple: one global GeneticAlgorithm object
in memory (fine for a single-user college project / demo).
"""

from flask import Flask, jsonify, render_template
from ai import GeneticAlgorithm

app = Flask(__name__)

# A single shared Genetic Algorithm instance.
# Recreated whenever /api/reset is called.
ga = GeneticAlgorithm(population_size=40)


@app.route("/")
def index():
    """Serve the main game page."""
    return render_template("index.html")


@app.route("/api/train_generation", methods=["POST"])
def train_generation():
    """
    Run ONE generation of the Genetic Algorithm.
    The frontend calls this repeatedly (once per generation) while
    the user is in "AI Mode", so the page can visually show the
    bird's behaviour improving generation after generation.
    """
    result = ga.run_generation()
    return jsonify(result)


@app.route("/api/best_weights", methods=["GET"])
def best_weights():
    """Return the best weights found so far, without training further."""
    return jsonify({
        "generation": ga.generation,
        "best_score_ever": ga.best_score_ever,
        "weights": ga.best_genome.weights,
    })


@app.route("/api/reset", methods=["POST"])
def reset():
    """Reset the AI: start learning again from generation 0."""
    ga.reset()
    return jsonify({"status": "reset", "generation": ga.generation})


if __name__ == "__main__":
    # debug=True is handy for a college project (auto-reload on save)
    app.run(debug=True, port=5000)
