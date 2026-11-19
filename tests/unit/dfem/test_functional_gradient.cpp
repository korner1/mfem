// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../unit_tests.hpp"

#include "mfem.hpp"

#ifdef MFEM_USE_MPI

#include "../../../fem/dfem/doperator.hpp"
#include "../../../fem/dfem/backends/global_qf/prelude.hpp"
#include "../../../fem/dfem/backends/local_qf/prelude.hpp"
#include "../../../linalg/tensor_arrays.hpp"

#include <array>
#include <cmath>
#include <type_traits>

using namespace mfem;
using namespace mfem::future;
using mfem::future::tensor;

namespace functional_gradient_test
{

// Dirichlet-type energy 1/2 \int |grad u|^2 dx. It depends on the solution and
// on the mesh coordinates, so both derivatives are nontrivial. The coordinate
// field enters only through Gradient<Coords>, while the solution enters
// through both Value<U> and Gradient<U>, so the two derivatives have a
// different number of outputs.
template <int dim>
struct EnergyFunctional
{
   MFEM_HOST_DEVICE inline MFEM_FORCE_INLINE
   auto operator()(const real_t &u,
                   const tensor<real_t, dim> &dudxi,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w,
                   real_t &f) const
   {
      const auto invJ = inv(J);
      const auto dudx = dudxi * invJ;
      f = (0.5_r * sqnorm(dudx) + 0.25_r * u * u) * det(J) * w;
   }
};

/// Energy functional with derivatives w.r.t. both the solution and the mesh
/// coordinates, plus a finite-difference reference for either one.
template <int dim>
class EnergyWithShapeDerivative
{
   static constexpr int U = 0, Coords = 1, Q = 2;

public:
   EnergyWithShapeDerivative(const ParFiniteElementSpace &fes,
                             const ParFiniteElementSpace &mfes,
                             const IntegrationRule &ir) :
      comm(fes.GetComm()),
      qspace(*fes.GetParMesh(), ir),
      qspace_vec(qspace, 1),
      q(qspace_vec)
   {
      const auto &pmesh = *fes.GetParMesh();
      Array<int> all_domain_attr;
      if (pmesh.attributes.Size() > 0)
      {
         all_domain_attr.SetSize(pmesh.attributes.Max());
         all_domain_attr = 1;
      }

      const auto in = std::vector
      {
         FieldDescriptor{U, &fes},
         FieldDescriptor{Coords, &mfes}
      };
      const auto out = std::vector
      {
         FieldDescriptor{Q, &qspace_vec}
      };

      dop = std::make_unique<DifferentiableOperator>(in, out, pmesh);
      EnergyFunctional<dim> energy;
      dop->AddDomainIntegrator<LocalQFBackend>(
         energy,
         Inputs<Value<U>, Gradient<U>, Gradient<Coords>, Weight> {},
         Outputs<FunctionalValue<Q>> {},
         ir, all_domain_attr, Derivatives<U, Coords> {});
   }

   real_t Eval(const Vector &u, const Vector &x) const
   {
      MultiVector X{u, x};
      MultiVector Y{q};
      dop->Mult(X, Y);
      const real_t local = q.Sum();
      real_t global;
      MPI_Allreduce(&local, &global, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                    comm);
      return global;
   }

   /// Gradient w.r.t. @a derivative_id, assembled into a Vector.
   void AssembleGradient(size_t derivative_id, const Vector &u,
                         const Vector &x, Vector &g) const
   {
      MultiVector X{u, x};
      dop->GetDerivative(derivative_id, X)->Assemble(g);
   }

   /// Central-difference directional derivative of the energy.
   real_t DirectionalFD(const Vector &u, const Vector &x,
                        const Vector &du, const Vector &dx,
                        real_t eps) const
   {
      Vector up(u), um(u), xp(x), xm(x);
      up.Add(eps, du);
      um.Add(-eps, du);
      xp.Add(eps, dx);
      xm.Add(-eps, dx);
      return (Eval(up, xp) - Eval(um, xm)) / (2.0_r * eps);
   }

   /// Global inner product of two T-vectors (true dofs are uniquely owned).
   real_t Dot(const Vector &a, const Vector &b) const
   {
      const real_t local = a * b;
      real_t global;
      MPI_Allreduce(&local, &global, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                    comm);
      return global;
   }

private:
   MPI_Comm comm;
   std::unique_ptr<DifferentiableOperator> dop;
   QuadratureSpace qspace;
   VectorQuadratureSpace qspace_vec;
   mutable QuadratureFunction q;
};

template <int DIM>
void functional_gradient(const char *filename, int p)
{
   CAPTURE(filename, DIM, p);

   static constexpr int U = 0, Coords = 1;

   Mesh smesh(filename);
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   pmesh.EnsureNodes();
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   ParFiniteElementSpace *mfes = nodes->ParFESpace();

   H1_FECollection fec(p, DIM);
   ParFiniteElementSpace fes(&pmesh, &fec);

   const IntegrationRule &ir =
      IntRules.Get(pmesh.GetTypicalElementGeometry(), 2 * p + 2);

   EnergyWithShapeDerivative<DIM> energy(fes, *mfes, ir);

   ParGridFunction u_gf(&fes), du_gf(&fes);
   FunctionCoefficient u_coeff(
   [](const auto &x) { return 1.0_r + x[0] + 0.25_r * x[1] * x[1]; });
   FunctionCoefficient du_coeff(
   [](const auto &x) { return sin(M_PI * x[0]) + 0.5_r * x[0] * x[1]; });
   u_gf.ProjectCoefficient(u_coeff);
   du_gf.ProjectCoefficient(du_coeff);

   ParGridFunction dx_gf(mfes);
   VectorFunctionCoefficient dx_coeff(
      DIM, [](const Vector &x, Vector &v)
   {
      v = 0.0;
      v(0) = 0.1_r * sin(M_PI * x[0]) * x[1];
      v(1) = 0.1_r * cos(M_PI * x[1]) * x[0];
   });
   dx_gf.ProjectCoefficient(dx_coeff);

   Vector u(fes.GetTrueVSize()), du(fes.GetTrueVSize());
   Vector x(mfes->GetTrueVSize()), dx(mfes->GetTrueVSize());
   u_gf.GetTrueDofs(u);
   du_gf.GetTrueDofs(du);
   nodes->GetTrueDofs(x);
   dx_gf.GetTrueDofs(dx);

   Vector zero_u(fes.GetTrueVSize()), zero_x(mfes->GetTrueVSize());
   zero_u = 0.0;
   zero_x = 0.0;

   const real_t eps = 1e-6;

   // Derivative w.r.t. the solution.
   Vector g_u;
   energy.AssembleGradient(U, u, x, g_u);
   REQUIRE(g_u.Size() == fes.GetTrueVSize());

   const real_t dJdu_fd = energy.DirectionalFD(u, x, du, zero_x, eps);
   REQUIRE(energy.Dot(g_u, du) == MFEM_Approx(dJdu_fd, 1e-6, 1e-8));

   // Derivative w.r.t. the mesh coordinates. This is the case that used to
   // assemble into the solution space instead of the coordinate space.
   Vector g_x;
   energy.AssembleGradient(Coords, u, x, g_x);
   REQUIRE(g_x.Size() == mfes->GetTrueVSize());

   const real_t dJdx_fd = energy.DirectionalFD(u, x, zero_u, dx, eps);
   REQUIRE(energy.Dot(g_x, dx) == MFEM_Approx(dJdx_fd, 1e-6, 1e-8));
}

template <typename scalar_t, int Dimension>
struct GlobalMaterialFunctional
{
   real_t scale = 1.0_r;

   void operator()(tensor_ndarray<const scalar_t, 2> &energies,
                   tensor_ndarray<const scalar_t, 2> &indicators,
                   tensor_ndarray<const real_t, 2> &densities,
                   tensor_array<const scalar_t, Dimension> &positions,
                   tensor_array<const scalar_t, Dimension, Dimension> &jacobians,
                   tensor_array<const real_t> &weights,
                   tensor_array<scalar_t> &output) const
   {
      const int num_mats = static_cast<int>(energies.size(0));
      const real_t energy_scale = scale;
      mfem::forall<UseEnzyme>(weights.size(), [=] MFEM_HOST_DEVICE(int point)
      {
         scalar_t total_energy{};
         for (int material = 0; material < num_mats; ++material)
         {
            const scalar_t energy = energies(material, point)();
            const scalar_t indicator = indicators(material, point)();
            total_energy += densities(material, point)() *
                            (indicator * (0.5_r * energy * energy +
                                          0.05_r * energy * energy * energy) +
                             0.125_r * indicator * indicator * energy);
         }
         output(point) = energy_scale * total_energy * det(jacobians(point)) *
                         weights(point)() * (1.0_r + 0.125_r * positions(point)(0));
      });
   }
};

template <int Dimension>
void global_material_functional(const char *filename, const int num_mats)
{
   CAPTURE(filename, Dimension, num_mats);

   static constexpr int Energy = 0, Indicators = 1, Density = 2;
   static constexpr int Coords = 3, Output = 4;
   using State = std::array<Vector, 4>;

   Mesh serial_mesh(filename);
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   mesh.EnsureNodes();
   auto *nodes = static_cast<ParGridFunction *>(mesh.GetNodes());
   H1_FECollection energy_collection(2, Dimension);
   L2_FECollection material_collection(1, Dimension);
   ParFiniteElementSpace energy_space(&mesh, &energy_collection, num_mats,
                                     Ordering::byVDIM);
   ParFiniteElementSpace material_space(&mesh, &material_collection, num_mats,
                                       Ordering::byVDIM);
   const std::array<ParFiniteElementSpace *, 4> spaces
   {
      &energy_space, &material_space, &material_space, nodes->ParFESpace()
   };
   const IntegrationRule &integration_rule =
      IntRules.Get(mesh.GetTypicalElementGeometry(), 6);
   QuadratureSpace quadrature_space(mesh, integration_rule);
   VectorQuadratureSpace output_space(quadrature_space, 1);
   QuadratureFunction output(output_space);
   const std::vector<FieldDescriptor> input_fields
   {
      {Energy, spaces[Energy]}, {Indicators, spaces[Indicators]},
      {Density, spaces[Density]}, {Coords, spaces[Coords]}
   };
   const std::vector<FieldDescriptor> output_fields{{Output, &output_space}};
   DifferentiableOperator functional(input_fields, output_fields, mesh);
   Array<int> attributes(mesh.attributes.Max());
   attributes = 1;
#ifdef MFEM_USE_ENZYME
   using scalar_t = real_t;
#else
   using scalar_t = dual<real_t, real_t>;
#endif
   GlobalMaterialFunctional<scalar_t, Dimension> quadrature_function;
   quadrature_function.scale = 1.37_r;
   using HessianBlocks = std::conditional_t<Dimension == 2,
         SecondDerivatives<DerivativePair<Energy, Energy>,
         DerivativePair<Energy, Indicators>>, SecondDerivatives<Pairs::None>>;
   functional.AddDomainIntegrator<GlobalQFBackend, DerivativeKernels::Action>(
      quadrature_function,
      Inputs<Value<Energy>, Value<Indicators>, Value<Density>,
      Value<Coords>, Gradient<Coords>, Weight> {}, Outputs<FunctionalValue<Output>> {},
      integration_rule, attributes, Derivatives<Energy, Indicators, Coords> {},
      HessianBlocks {});
   DifferentiableOperator local_functional(input_fields, output_fields, mesh);
   local_functional.SetMultLevel(DifferentiableOperator::LVECTOR);
   local_functional.AddDomainIntegrator<GlobalQFBackend, DerivativeKernels::Action>(
      quadrature_function,
      Inputs<Value<Energy>, Value<Indicators>, Value<Density>,
      Value<Coords>, Gradient<Coords>, Weight> {}, Outputs<FunctionalValue<Output>> {},
      integration_rule, attributes, Derivatives<Energy, Indicators, Coords> {});

   State state, directions;
   for (int field_id : {Energy, Indicators, Density})
   {
      const real_t scale = field_id == Indicators ? 1.0_r / num_mats : 1.0_r;
      VectorFunctionCoefficient coefficient(num_mats,
      [num_mats, scale](const Vector &position, Vector &values)
      {
         for (int material = 0; material < num_mats; ++material)
         {
            values(material) = scale * (1.0_r + 0.2_r * material +
                                        0.1_r * position(0) +
                                        0.05_r * position(1) * position(1));
         }
      });
      ParGridFunction field(spaces[field_id]);
      field.ProjectCoefficient(coefficient);
      field.GetTrueDofs(state[field_id]);
   }
   nodes->GetTrueDofs(state[Coords]);
   for (int field_id : {Energy, Indicators, Density, Coords})
   {
      directions[field_id].SetSize(spaces[field_id]->GetTrueVSize());
      directions[field_id].Randomize(71 + 13 * field_id + mesh.GetMyRank());
      directions[field_id] -= 0.5_r;
   }
   directions[Coords] *= 0.05_r;

   auto global_sum = [&](const real_t local)
   {
      real_t global;
      MPI_Allreduce(&local, &global, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                    mesh.GetComm());
      return global;
   };
   auto evaluate = [&](const State &values)
   {
      functional.SetMultLevel(DifferentiableOperator::TVECTOR);
      MultiVector input{values[Energy], values[Indicators], values[Density],
                        values[Coords]};
      MultiVector result{output};
      functional.Mult(input, result);
      return global_sum(output.Sum());
   };
   auto assemble = [&](const int field_id, const State &values, Vector &gradient)
   {
      functional.SetMultLevel(DifferentiableOperator::TVECTOR);
      MultiVector input{values[Energy], values[Indicators], values[Density],
                        values[Coords]};
      functional.GetDerivative(field_id, input)->Assemble(gradient);
   };
   auto norm = [&](const Vector &values)
   {
      return std::sqrt(InnerProduct(mesh.GetComm(), values, values));
   };
   const real_t epsilon = 1e-6_r;

   auto check_states = [&](const bool check_hessian)
   {
      for (int state_index = 0; state_index < 2; ++state_index)
      {
         CAPTURE(state_index);
         if (state_index != 0)
         {
            for (int field_id : {Energy, Indicators, Coords})
            {
               state[field_id].Add(0.1_r, directions[field_id]);
            }
            state[Density] *= 1.25_r;
         }
         functional.SetMultLevel(DifferentiableOperator::TVECTOR);
         MultiVector input{state[Energy], state[Indicators], state[Density],
                           state[Coords]};

         if (check_hessian)
         {
            for (int direction_id : {Energy, Indicators})
            {
               CAPTURE(direction_id);
               REQUIRE(functional.HasSecondDerivative(Energy, direction_id));
               auto hessian = functional.GetSecondDerivative(Energy, direction_id,
                                                             input);
               REQUIRE(hessian->Height() == spaces[Energy]->GetTrueVSize());
               REQUIRE(hessian->Width() == spaces[direction_id]->GetTrueVSize());
               Vector action(spaces[Energy]->GetTrueVSize());
               action = 7.0_r;
               MultiVector action_output{action};
               hessian->Mult(directions[direction_id], action_output);
               REQUIRE(norm(action) > 1e-8_r);

               State plus(state), minus(state);
               plus[direction_id].Add(epsilon, directions[direction_id]);
               minus[direction_id].Add(-epsilon, directions[direction_id]);
               Vector gradient_plus, gradient_minus;
               assemble(Energy, plus, gradient_plus);
               assemble(Energy, minus, gradient_minus);
               REQUIRE(gradient_plus.Size() == action.Size());
               REQUIRE(gradient_minus.Size() == action.Size());
               gradient_plus -= gradient_minus;
               gradient_plus /= 2.0_r * epsilon;
               gradient_plus -= action;
               REQUIRE(norm(gradient_plus) <= 2e-6_r * (1.0_r + norm(action)));

               Vector repeated(action.Size());
               repeated = -3.0_r;
               MultiVector repeated_output{repeated};
               hessian->Mult(directions[direction_id], repeated_output);
               repeated -= action;
               REQUIRE(norm(repeated) <= 1e-12_r * (1.0_r + norm(action)));
            }
         }
         else
         {
            const real_t value = evaluate(state);
            REQUIRE(value > 0.0_r);
            State scaled(state);
            scaled[Density] *= 1.5_r;
            REQUIRE(evaluate(scaled) == MFEM_Approx(1.5_r * value));

            State local_state;
            for (int field_id : {Energy, Indicators, Density, Coords})
            {
               local_state[field_id].SetSize(spaces[field_id]->GetVSize());
               spaces[field_id]->GetProlongationMatrix()->Mult(
                  state[field_id], local_state[field_id]);
            }
            MultiVector local_input{local_state[Energy], local_state[Indicators],
                                    local_state[Density], local_state[Coords]};
            MultiVector result{output};
            local_functional.Mult(local_input, result);
            REQUIRE(global_sum(output.Sum()) == MFEM_Approx(value));

            for (int field_id : {Energy, Indicators, Coords})
            {
               CAPTURE(field_id);
               functional.SetMultLevel(DifferentiableOperator::TVECTOR);
               auto derivative = functional.GetDerivative(field_id, input);
               Vector gradient(1);
               gradient = 7.0_r;
               derivative->Assemble(gradient);
               REQUIRE(gradient.Size() == spaces[field_id]->GetTrueVSize());

               State plus(state), minus(state);
               plus[field_id].Add(epsilon, directions[field_id]);
               minus[field_id].Add(-epsilon, directions[field_id]);
               const real_t finite_difference =
                  (evaluate(plus) - evaluate(minus)) / (2.0_r * epsilon);
               REQUIRE(InnerProduct(mesh.GetComm(), gradient, directions[field_id])
                       == MFEM_Approx(finite_difference, 1e-7, 1e-6));

               Vector repeated(gradient.Size() + 1);
               repeated = -3.0_r;
               derivative->Assemble(repeated);
               REQUIRE(repeated.Size() == spaces[field_id]->GetTrueVSize());
               repeated -= gradient;
               REQUIRE(norm(repeated) <= 1e-12_r * (1.0_r + norm(gradient)));

               Vector local_gradient;
               local_functional.GetDerivative(field_id, local_input)->Assemble(
                  local_gradient);
               REQUIRE(local_gradient.Size() == spaces[field_id]->GetVSize());
               Vector assembled(spaces[field_id]->GetTrueVSize());
               spaces[field_id]->GetProlongationMatrix()->MultTranspose(
                  local_gradient, assembled);
               assembled -= gradient;
               REQUIRE(norm(assembled) <= 1e-12_r * (1.0_r + norm(gradient)));
            }
         }
      }
   };

   SECTION("gradient assembly")
   {
      check_states(false);
   }
   if constexpr (Dimension == 2)
   {
      SECTION("material Hessian actions")
      {
         check_states(true);
      }
   }
}

} // namespace functional_gradient_test

#ifdef MFEM_USE_ENZYME
TEST_CASE("dFEM functional gradient assembles into a Vector",
          "[Parallel][dFEM][functional-gradient]")
{
   const bool all_tests = launch_all_non_regression_tests;
   const auto p = !all_tests ? 1 : GENERATE(1, 2, 3);

   SECTION("2d")
   {
      const auto f = GENERATE("../../data/inline-quad.mesh",
                              "../../data/star.mesh");
      functional_gradient_test::functional_gradient<2>(f, p);
   }

   SECTION("3d")
   {
      const auto f = GENERATE("../../data/inline-hex.mesh");
      functional_gradient_test::functional_gradient<3>(f, p);
   }
}

#endif

TEST_CASE("dFEM native global functional with runtime materials",
          "[Parallel][dFEM][functional-gradient][global-qf][runtime-materials]")
{
   const int num_mats = GENERATE(1, 3, 5);

   SECTION("2d")
   {
      functional_gradient_test::global_material_functional<2>(
         "../../data/inline-quad.mesh", num_mats);
   }

   SECTION("3d")
   {
      functional_gradient_test::global_material_functional<3>(
         "../../data/inline-hex.mesh", num_mats);
   }
}

#endif // MFEM_USE_MPI
