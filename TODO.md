# TODO for code cleanup

**functionality**
- add custom DDPM scheduler based on diffusers DDPM scheduler
- add hugging face diffusers to pyproject toml (how to correctly install it such that torch works?)
- diffusion model should be passed a noise process object, and not individual ones


**chores, cleanup**
- tidy up basically all files (apart maybe from sde_lib) with best practices

**Tests**  
- check if inpainting works with the new implementation

TODO: remove todos before merging