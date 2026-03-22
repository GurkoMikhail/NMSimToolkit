1. Update `core/transport/simulation_managers_soa.py`.
2. Import `SourceSoA` from `core.source.sources_soa` and update `source` type annotation from `Any` to `SourceSoA`.
3. In `__init__`, fix the instantiation of `ParticleBank` from `ParticleBank.allocate(self.particles_number)` to `ParticleBank(self.particles_number)`.
4. In `_run()`, replace `self.bank.inject_particles(self.source.generate_particles(self.particles_number))` with `self.source.inject(self.bank, self.particles_number)`.
5. In `next_step()`, under the "Continuous Replenishment" block, replace:
   ```python
   new_particles = self.source.generate_particles(num_to_inject)
   self.bank.inject_particles(new_particles)
   ```
   with:
   ```python
   self.source.inject(self.bank, num_to_inject)
   ```
6. Run `pytest tests/transport/test_simulation_managers_soa.py` (or similar relevant tests if they exist) and root-level tests as per pre commit steps.
7. Submit the changes.
