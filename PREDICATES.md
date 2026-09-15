# Predicate Definitions

This document is the repository specification for the **83 semantic predicates** used in the paper *Grounding Vision–Language Models in Driving Semantics: A Multi-Dataset Predicate Framework for Explainable Reasoning*. It is organized by predicate families including operational definitions, equations, applicability conditions, and the used numerical thresholds.

Predicate names are materialised in the repository with the `np:` namespace (for example, paper predicate `follows` is emitted as `np:follows`).

**Inventory:** 12 spatial + 17 motion + 17 temporal + 25 map + 8 interaction + 3 traffic-control + 1 risk = **83 semantic predicates**.

---

Throughout,

$$
P\mathrel{:\!\Leftrightarrow}C
$$

denotes a definitional equivalence: predicate $P$ is defined to hold exactly when condition $C$ is satisfied. Ordinary equalities are used for stored numerical quantities. Numerical thresholds are stated with the corresponding predicate definitions or, for safety-related criteria, in the Safety-Related Thresholds section below.

## Common Notation

Let $E$ denote a spatial entity, $S$ the subject of a directed pair, and $O$ the object. Their planar centers are $\mathbf{p}_S=[x_S,y_S]^\top$ and $\mathbf{p}_O=[x_O,y_O]^\top$, with subject body heading $\theta_S$. The relative position is

$$
\Delta\mathbf{p}_{SO}
    =\mathbf{p}_O-\mathbf{p}_S=[dx,dy]^\top .
$$

The subject-forward and subject-left unit vectors are

$$
\mathbf{e}_S=
\begin{bmatrix}\cos\theta_S\\\sin\theta_S\end{bmatrix},
\qquad
\mathbf{n}_S=
\begin{bmatrix}-\sin\theta_S\\\cos\theta_S\end{bmatrix}.
$$

The body-frame longitudinal and lateral displacements are

$$
\begin{aligned}
    l_{SO}&=\mathbf{e}_S^\top\Delta\mathbf{p}_{SO},\\
    r_{SO}&=\mathbf{n}_S^\top\Delta\mathbf{p}_{SO},
\end{aligned}
$$

so that $l_{SO}>0$ means ahead and $r_{SO}>0$ means left. The center distance is

$$
d_c(S,O)=\lVert\Delta\mathbf{p}_{SO}\rVert_2.
$$

Let $F_E$ be the oriented rectangular footprint of entity $E$. The free-space clearance and intersection area of a pair are

$$
\begin{aligned}
d_{fs}(S,O)&=\operatorname{dist}(F_S,F_O),\\
A_\cap(S,O)&=\operatorname{area}(F_S\cap F_O).
\end{aligned}
$$

Let $\mathbf{v}_S$ and $\mathbf{v}_O$ be their global planar velocity vectors and

$$
\Delta\mathbf{v}_{SO}=\mathbf{v}_O-\mathbf{v}_S
    =[\Delta v_x,\Delta v_y]^\top.
$$

The operator $\operatorname{wrap}(\alpha)$ maps an angle to $[-\pi,\pi)$. For temporal predicates, $k$ indexes an observation and $t_k$ denotes its timestamp. Dataset-specific timestamps are normalised to seconds before predicate evaluation, such that

$$
\Delta t_k=t_k-t_{k-1}
$$

is expressed in seconds. With the default sample interval $T_s=0.5\,\mathrm{s}$ and maximum temporal-gap factor $1.5$, continuity is preserved only when $\Delta t_k\leq0.75\,\mathrm{s}$.

## Spatial Predicates

The eight directional predicates are classified in the subject body-heading frame. The defaults are a longitudinal deadband $\tau_l=1.0\,\mathrm{m}$, lateral deadband $\tau_r=0.75\,\mathrm{m}$, pure-left/right band $\tau_c=2.0\,\mathrm{m}$, and front/rear corridor

$$
w(l)=2.5+0.30|l|\quad[\mathrm{m}].
$$

Exactly one primary directional sector is selected when the pair geometry is valid.

#### `inFrontOf`

$$
\begin{aligned}
\texttt{inFrontOf}(S,O)\mathrel{:\!\Leftrightarrow}{}& l_{SO}\geq1\land\Bigl[\\
&(|l_{SO}|\leq2\land |r_{SO}|<0.75)\\
&\lor\bigl(|l_{SO}|>2\\
&\qquad\land |r_{SO}|\leq w(l_{SO})\bigr)\Bigr].
\end{aligned}
$$

#### `behind`

$$
\begin{aligned}
\texttt{behind}(S,O)\mathrel{:\!\Leftrightarrow}{}& l_{SO}\leq-1\land\Bigl[\\
&(|l_{SO}|\leq2\land |r_{SO}|<0.75)\\
&\lor\bigl(|l_{SO}|>2\\
&\qquad\land |r_{SO}|\leq w(l_{SO})\bigr)\Bigr].
\end{aligned}
$$

#### `leftOf` and `rightOf`

$$
\begin{aligned}
\texttt{leftOf}(S,O)
&\mathrel{:\!\Leftrightarrow}|l_{SO}|\leq2\land r_{SO}\geq0.75,\\
\texttt{rightOf}(S,O)
&\mathrel{:\!\Leftrightarrow}|l_{SO}|\leq2\land r_{SO}\leq-0.75.
\end{aligned}
$$

#### `frontLeftOf` and `frontRightOf`

$$
\begin{aligned}
\texttt{frontLeftOf}(S,O)\mathrel{:\!\Leftrightarrow}{}&l_{SO}>2\\
&\land r_{SO}>w(l_{SO}),\\
\texttt{frontRightOf}(S,O)\mathrel{:\!\Leftrightarrow}{}&l_{SO}>2\\
&\land r_{SO}<-w(l_{SO}).
\end{aligned}
$$

#### `rearLeftOf` and `rearRightOf`

$$
\begin{aligned}
\texttt{rearLeftOf}(S,O)\mathrel{:\!\Leftrightarrow}{}&l_{SO}<-2\\
&\land r_{SO}>w(l_{SO}),\\
\texttt{rearRightOf}(S,O)\mathrel{:\!\Leftrightarrow}{}&l_{SO}<-2\\
&\land r_{SO}<-w(l_{SO}).
\end{aligned}
$$

#### `overlapping`

With $\varepsilon_A=10^{-4}\,\mathrm{m}^2$,

$$
\texttt{overlapping}(S,O)
    \mathrel{:\!\Leftrightarrow}A_\cap(S,O)>\varepsilon_A.
$$

#### `touching`

With geometric distance tolerance $\varepsilon_d=10^{-3}\,\mathrm{m}$,

$$
\begin{aligned}
\texttt{touching}(S,O)\mathrel{:\!\Leftrightarrow}{}&A_\cap(S,O)\leq\varepsilon_A\\
&\land\Bigl(F_S\text{ touches }F_O\\
&\qquad\lor d_{fs}\leq\varepsilon_d\Bigr).
\end{aligned}
$$

#### `veryNear` and `near`

$$
\begin{aligned}
\texttt{veryNear}(S,O)
&\mathrel{:\!\Leftrightarrow}0<d_{fs}(S,O)\leq2\,\mathrm{m},\\
\texttt{near}(S,O)
&\mathrel{:\!\Leftrightarrow}2<d_{fs}(S,O)\leq5\,\mathrm{m}.
\end{aligned}
$$

The four distance/contact states are mutually exclusive.

## Motion Predicates

For an entity $E$, let $\mathbf{v}_E=[v_x,v_y]^\top$. Tracked-object velocities are interpreted in the common global frame. Ego velocity is converted from the ego-state representation to the global geometric-center velocity used by the predicate framework.

#### `hasVelocity`, `hasVelocityX`, `hasVelocityY`, and `hasSpeed`

The planar velocity of entity $E$ is

$$
\mathbf{v}_E=[v_x,v_y]^\top .
$$

The predicate `hasVelocity` stores this planar velocity, while `hasVelocityX` and `hasVelocityY` store its individual components. The corresponding speed magnitude is

$$
v_E=\sqrt{v_x^2+v_y^2}.
$$

#### `hasAccelerationX`, `hasAccelerationY`, and `hasAcceleration`

For ego states exposing native planar acceleration,

$$
a_E=\sqrt{a_x^2+a_y^2}.
$$

The component predicates store $a_x$ and $a_y$.

#### `hasRelativeSpeedTo`

The relative-speed magnitude is

$$
v^{rel}_{SO}=\lVert\Delta\mathbf{v}_{SO}\rVert_2.
$$

#### `hasLongitudinalRelativeSpeedTo`

The subject-longitudinal relative velocity is

$$
v^{rel,long}_{SO}
    =\mathbf{e}_S^\top\Delta\mathbf{v}_{SO}.
$$

#### `hasLateralRelativeSpeedTo`

The subject-lateral relative velocity is

$$
v^{rel,lat}_{SO}
    =\mathbf{n}_S^\top\Delta\mathbf{v}_{SO}.
$$

#### `hasClosingSpeedTo`

The radial closing speed is

$$
v^{close}_{SO} =
-\frac{\Delta\mathbf{p}_{SO}^{\top}\Delta\mathbf{v}_{SO}}
       {\lVert\Delta\mathbf{p}_{SO}\rVert_2},
\qquad d_c(S,O)>0.
$$

Positive values mean decreasing center separation.

#### `hasVelocityTowardTarget`

The component of subject velocity toward the current object center is

$$
v^{\rightarrow O}_{S} =
\frac{\mathbf{v}_S^\top\Delta\mathbf{p}_{SO}}
     {\lVert\Delta\mathbf{p}_{SO}\rVert_2},
\qquad d_c(S,O)>0.
$$

#### `hasSubjectForwardSpeed`

The subject forward speed in its body-heading frame is

$$
v_S^{fwd}=\mathbf{e}_S^\top\mathbf{v}_S.
$$

#### `hasVelocityHeading`

When $v_E\geq0.75\,\mathrm{m/s}$, the velocity heading is

$$
\theta_E^v=
    \operatorname{wrap}\!\left(\operatorname{atan2}(v_y,v_x)\right).
$$

Below this speed the velocity direction is considered insufficiently stable and the predicate is not emitted.

#### `hasEffectiveTravelHeading`

This predicate stores a conservative travel direction selected from map heading, velocity heading, and displacement heading. A displacement heading is considered informative only after at least $0.40\,\mathrm{m}$ displacement; a velocity heading requires at least $0.75\,\mathrm{m/s}$. If both motion cues are available, they must agree within $0.45\,\mathrm{rad}$. An accepted motion-derived direction must agree with an available unambiguous map heading within $0.60\,\mathrm{rad}$. The selected angle is denoted $\theta_E^{travel}$.

#### `hasTravelDirectionSource`

Stores the categorical provenance of $\theta_E^{travel}$, i.e., which accepted map/motion evidence supplied the selected direction.

#### `hasTravelDirectionDifferenceTo`

When both participants have effective travel headings,

$$
\Delta\theta^{travel}_{SO} =
\left|
\operatorname{wrap}
(\theta_O^{travel}-\theta_S^{travel})
\right|.
$$

## Temporal Predicates

The continuity limit is

$$
T_{gap}=0.5\times1.5=0.75\,\mathrm{s}.
$$

Temporal derivatives and continuous-streak quantities are not bridged across larger gaps.

#### `precedes`

For two instances of the same persistent track,

$$
\begin{aligned}
\texttt{precedes}(E_{k-1},E_k)\mathrel{:\!\Leftrightarrow}{}&
\operatorname{sameTrack}(E_{k-1},E_k)\\
&\land\,0<\Delta t_k\leq0.75\,\mathrm{s}.
\end{aligned}
$$

#### `hasDeltaTimeFromPrevious`

Stores $\Delta t_k$ in seconds for a valid preceding observation.

#### `hasDisplacementFromPrevious`

$$
d_k^{disp}=\lVert\mathbf{p}_k-\mathbf{p}_{k-1}\rVert_2.
$$

#### `hasHeadingChangeFromPrevious`

$$
\Delta\theta_k
    =\operatorname{wrap}(\theta_k-\theta_{k-1}).
$$

#### `hasSpeedChangeFromPrevious`

$$
\Delta v_k=v_k-v_{k-1}.
$$

#### `hasEstimatedAcceleration`

For $\Delta t_k>0$,

$$
\hat a_k=\frac{\Delta v_k}{\Delta t_k}.
$$

#### `hasDisplacementHeading`

When $d_k^{disp}\geq0.40\,\mathrm{m}$,

$$
\theta_k^{disp} =
\operatorname{atan2}(y_k-y_{k-1},x_k-x_{k-1}).
$$

#### `hasObservedFrameCount` and `hasObservedDuration`

These predicates store the count and elapsed duration of the entity history maintained by the current scenario processing state. The duration is measured from the first available observation in that maintained history to the current timestamp.

#### `hasTotalObservedFrameCount` and `hasTotalObservedSpan`

The total count includes all available entity observations in the current processing window, including observations separated by gaps. If $t_{first}$ is the first observation timestamp,

$$
T_E^{total}=t_k-t_{first}.
$$

#### `hasContinuousObservedFrameCount` and `hasContinuousObservedDuration`

The continuous count increments only while successive observations satisfy the $0.75\,\mathrm{s}$ continuity condition. When a gap exceeds that limit the streak restarts at one. If $t_{cont,0}$ is the start of the current streak,

$$
T_{E,k}^{cont}=t_k-t_{cont,0}.
$$

#### `hasPairObservedFrameCount` and `hasPairObservedDuration`

The same continuity logic is maintained for each directed pair. If $t_{pair,0}$ is the first timestamp of the current uninterrupted co-observation streak,

$$
T_{SO,k}^{pair}=t_k-t_{pair,0}.
$$

#### `hasCenterDistanceChangeFromPrevious`

$$
\Delta d_{c,k}=d_{c,k}-d_{c,k-1}.
$$

#### `hasFreeSpaceDistanceChangeFromPrevious`

$$
\Delta d_{fs,k}=d_{fs,k}-d_{fs,k-1}.
$$

## Map Predicates

Let $M$ denote an HD-map polygon and let

$$
\rho(E,M) =
\frac{\operatorname{area}(F_E\cap M)}
     {\operatorname{area}(F_E)}
$$

be footprint overlap. Primary-map matching uses exact unbuffered map geometry. A candidate must contain the entity center or provide sufficient footprint overlap. The primary-map configuration uses a minimum overlap ratio $0.20$, an ambiguity margin $0.05$, and a lateral tie-break margin $0.50\,\mathrm{m}$.

#### `inLane` and `inLaneConnector`

Center-membership predicates are denoted below by $I_L$ and $I_C$, respectively:

$$
\begin{aligned}
I_L(E,L)&\mathrel{:\!\Leftrightarrow}\mathbf{p}_E\in\operatorname{polygon}(L),\\
I_C(E,C)&\mathrel{:\!\Leftrightarrow}\mathbf{p}_E\in\operatorname{polygon}(C).
\end{aligned}
$$

#### `intersectsLane` and `intersectsLaneConnector`

Writing $J_L$ and $J_C$ for the corresponding footprint-intersection predicates,

$$
\begin{aligned}
J_L(E,L)&\mathrel{:\!\Leftrightarrow}\operatorname{area}(F_E\cap L)>0,\\
J_C(E,C)&\mathrel{:\!\Leftrightarrow}\operatorname{area}(F_E\cap C)>0.
\end{aligned}
$$

#### `hasPrimaryLane` and `hasPrimaryLaneConnector`

These predicates store the unique conservative primary lane $L^{*}$ or lane connector $C^{*}$ selected from the exact candidate set. When competing candidates are too similar under the overlap/center/lateral/heading evidence, no forced primary assignment is made.

#### `hasPrimaryMapOverlapRatio`

For the selected primary primitive $M^{*}$,

$$
\rho_E^{*}=\rho(E,M^{*}).
$$

#### `hasAmbiguousMapMatch`

A positive Boolean indicator is emitted when candidate evidence is too ambiguous for a reliable primary map assignment under the $0.05$ ambiguity margin and associated deterministic tie-breaking logic.

#### `hasBaselineProgress`

If $p_E^{*}$ is the nearest point on the selected directed baseline and $s(\cdot)$ its arc-length coordinate,

$$
s_E=s(p_E^{*}).
$$

#### `hasBaselineLateralOffset`

For baseline pose $(x_b,y_b,\theta_b)$ at $p_E^{*}$,

$$
r_E^{map} =
-\sin\theta_b(x_E-x_b)
+\cos\theta_b(y_E-y_b).
$$

Positive values lie left of the directed baseline.

#### `hasMapHeading`

Stores the local legal travel direction

$$
\theta_E^{map}=\operatorname{wrap}(\theta_b).
$$

#### `hasBaselineCurvature`

Stores the local signed baseline curvature

$$
\kappa_E=\kappa(s_E).
$$

#### `hasMapSpeedLimit`

Stores the speed limit of $M^{*}$ in $\mathrm{m/s}$ when the map primitive provides a finite valid value.

#### `hasParentRoadblock` and `hasParentRoadblockConnector`

These relations expose the parent roadblock of a selected lane and the parent roadblock connector of a selected lane connector, respectively.

#### `inIntersection` and `intersectsIntersection`

Writing $I_{int}$ and $J_{int}$ for center and footprint intersection,

$$
\begin{aligned}
I_{int}(E,I)&\mathrel{:\!\Leftrightarrow}\mathbf{p}_E\in\operatorname{polygon}(I),\\
J_{int}(E,I)&\mathrel{:\!\Leftrightarrow}\operatorname{area}(F_E\cap I)>0.
\end{aligned}
$$

#### `hasPrimaryMapIntersection`

Stores the intersection topologically/geometrically associated with the selected primary lane or connector.

#### `inCrosswalk` and `intersectsCrosswalk`

Writing $I_{cw}$ and $J_{cw}$ for center and footprint crosswalk membership,

$$
\begin{aligned}
I_{cw}(E,CW)&\mathrel{:\!\Leftrightarrow}\mathbf{p}_E\in\operatorname{polygon}(CW),\\
J_{cw}(E,CW)&\mathrel{:\!\Leftrightarrow}\operatorname{area}(F_E\cap CW)>0.
\end{aligned}
$$

#### `hasSpatialMapRelation`

For reliable primary map primitives $M_S^{*}$ and $M_O^{*}$, this categorical predicate stores whether they are the same primitive, left/right adjacent, successor/predecessor, or unrelated according to the directed map topology.

#### `hasMapProgressDifferenceTo`

When both entities share the same selected baseline,

$$
\Delta s_{SO}=s_O-s_S.
$$

#### `hasSignedPathDistanceTo`

On the same primary primitive,

$$
d_{SO}^{path}=s_O-s_S.
$$

If $O$ lies downstream on an accepted direct/connected successor path, the remaining arc length on the subject primitive and downstream progress are summed. Predecessor paths receive the corresponding negative sign. The search does not force a path through an ambiguous branch.

#### `inSameLaneAs`

$$
\begin{aligned}
\texttt{inSameLaneAs}(S,O)\mathrel{:\!\Leftrightarrow}{}&
L_S^{*}=L_O^{*}\\
&\land\neg A_S^{map}\land\neg A_O^{map},
\end{aligned}
$$

where $A_E^{map}$ is the map-ambiguity flag. Travel-direction agreement is not required by this predicate.

#### `sharesIntersectionWith`

Asserted when both entities have independently established geometric or topological association with the same intersection.

## Safety-Related Thresholds

Safety-related thresholds for longitudinal following, collision risk, and lane-change gaps follow the criteria described in UN Regulation No. 157 (UN R157). For M1/N1 vehicles travelling at speed $v\leq60\,\mathrm{km/h}$, the minimum following distance is

$$
d_{\min}(v)=v\,t_{\mathrm{front}}(v),
$$

where $v$ is expressed in $\mathrm{m/s}$. The minimum time-gap values are

| $v$ [km/h] | 7.2 | 10 | 20 | 30 | 40 | 50 | 60 |
|:-----------:|:---:|:--:|:--:|:--:|:--:|:--:|:--:|
| $t_{\mathrm{front}}$ [s] | 1.0 | 1.1 | 1.2 | 1.3 | 1.4 | 1.5 | 1.6 |

with piecewise-linear interpolation between the specified values. For speeds below $2\,\mathrm{m/s}$, a minimum following distance of $2\,\mathrm{m}$ applies.

An imminent collision risk is identified when collision avoidance would require a braking demand of at least $5\,\mathrm{m/s^2}$. For regular lane changes, the approaching vehicle in the target lane shall not be required to decelerate by more than $3.0\,\mathrm{m/s^2}$ under the corresponding assessment conditions, and the inter-vehicle distance shall not become smaller than the distance travelled by the ALKS vehicle in $1.0\,\mathrm{s}$.

## Interaction Predicates

Interaction predicates are deterministic temporal relations that combine geometric, kinematic, map, and history evidence. The auxiliary condition symbols below are abbreviations for the explicitly defined condition blocks.

### `follows`

For a participant $X$ projected onto a directed lane/connector path, let $I_X=[s_X^{rear},s_X^{front}]$ denote its footprint interval and

$$
g_{SO}=s_O^{rear}-s_S^{front}
$$

the forward bumper gap. The complete relation is

$$
\begin{aligned}
\texttt{follows}(S,O)\mathrel{:\!\Leftrightarrow}{}&
\mathcal F_{\mathrm{veh}}
\land\mathcal F_{\mathrm{path}}
\land\mathcal F_{\mathrm{leader}}\\
&\land
(\mathcal F_{\mathrm{move}}\lor\mathcal F_{\mathrm{queue}})
\land\mathcal F_{\mathrm{persist}} .
\end{aligned}
$$

$\mathcal F_{\mathrm{veh}}$ requires compatible motor-vehicle participants. $\mathcal F_{\mathrm{path}}$ requires one reliable unambiguous directed lane/connector path, at most three topology hops, positive gap, non-overlapping footprints, and no reverse path motion beyond $0.30\,\mathrm{m/s}$. $\mathcal F_{\mathrm{leader}}$ requires $O$ to be the unique nearest valid forward leader; candidates within $0.50\,\mathrm{m}$ of one another are treated as ambiguous.

For moving traffic,

$$
\begin{aligned}
\mathcal F_{\mathrm{move}}\mathrel{:\!\Leftrightarrow}{}&
v_S^{path}>0.30\,\mathrm{m/s}\\
&\land 0<g_{SO}/v_S^{path}\leq5.0\,\mathrm{s}\\
&\land g_{SO}\leq80\,\mathrm{m},
\end{aligned}
$$

where $v_S^{path}$ is the subject velocity projected onto the directed path. For stop-and-go traffic,

$$
\begin{aligned}
\mathcal F_{\mathrm{queue}}\mathrel{:\!\Leftrightarrow}{}&
v_S^{path}\leq2.0\,\mathrm{m/s}\\
&\land v_O^{path}\leq4.0\,\mathrm{m/s}\\
&\land 0<g_{SO}\leq12\,\mathrm{m}.
\end{aligned}
$$

Finally,

$$
\mathcal F_{\mathrm{persist}}
    \mathrel{:\!\Leftrightarrow}T_{\mathrm{cond}}\geq1.0\,\mathrm{s}.
$$

Thus, queue following is included in `follows`, while the queue-specific case is additionally represented by the separate `queuesBehind` predicate.

### `queuesBehind`

The predicate `queuesBehind` represents the queue-specific case of longitudinal following. It uses the same vehicle, path, leader, and persistence requirements as `follows`, while requiring the stop-and-go condition:

$$
\begin{aligned}
\texttt{queuesBehind}(S,O)\mathrel{:\!\Leftrightarrow}{}&
\mathcal F_{\mathrm{veh}}
\land\mathcal F_{\mathrm{path}}
\land\mathcal F_{\mathrm{leader}}\\
&\land\mathcal F_{\mathrm{queue}}
\land\mathcal F_{\mathrm{persist}} .
\end{aligned}
$$

### `changesLane`

Let $C_s$ and $C_t$ denote the stable decoded source and target logical lane corridors and let $L_t$ be the emitted target lane. The implemented event can be summarized as

$$
\begin{aligned}
\texttt{changesLane}(S,L_t)\mathrel{:\!\Leftrightarrow}{}&
\mathcal L_{\mathrm{hist}}
\land\mathcal L_{\mathrm{trans}}
\land\mathcal L_{\mathrm{motion}}\\
&\land\mathcal L_{\mathrm{target}}
\land\mathcal L_{\mathrm{score}} .
\end{aligned}
$$

The history gate is

$$
\mathcal L_{\mathrm{hist}}
\mathrel{:\!\Leftrightarrow}
N_{\mathrm{hist}}\geq1
\land T_{\mathrm{hist}}\geq0.5\,\mathrm{s}.
$$

The transition must be lateral rather than an ordinary longitudinal lane–connector continuation:

$$
\begin{aligned}
\mathcal L_{\mathrm{trans}}\mathrel{:\!\Leftrightarrow}{}&
C_s\neq C_t
\land\operatorname{Adjacent}(C_s,C_t)\\
&\land\neg\operatorname{Continuation}(C_s,C_t).
\end{aligned}
$$

The principal physical gate is

$$
\begin{aligned}
\mathcal L_{\mathrm{motion}}\mathrel{:\!\Leftrightarrow}{}&
v_S\geq0.10\,\mathrm{m/s}\\
&\land T_{\mathrm{trans}}\leq8.0\,\mathrm{s}\\
&\land|\Delta r|\geq1.0\,\mathrm{m}.
\end{aligned}
$$

The target normally requires two stable frames. Stable-lane evidence uses overlap ratio at least $0.45$. Physical onset uses lateral offset $0.15\,\mathrm{m}$, lateral velocity $0.15\,\mathrm{m/s}$, future target gain $0.40\,\mathrm{m}$, and two onset-support frames within a $6.0\,\mathrm{s}$ search window. Completion uses target overlap at least $0.60$, source remaining overlap at most $0.40$, and two stable completion frames; the enabled high-confidence single-frame completion requires target overlap at least $0.80$. A single-frame primary-target fallback requires overlap at least $0.70$.

Official map adjacency is preferred. The geometric fallback allows polygon separation at most $1.5\,\mathrm{m}$, centerline distance $2.0$–$6.5\,\mathrm{m}$, and direction difference at most $0.30\,\mathrm{rad}$; the local fallback uses centerline distance $1.25$–$6.5\,\mathrm{m}$ and direction difference at most $0.60\,\mathrm{rad}$. The decoder permits at most three unknown-gap frames and requires

$$
\begin{aligned}
\mathcal L_{\mathrm{score}}\mathrel{:\!\Leftrightarrow}{}&
(s_{\mathrm{LC}}\geq6.0
\land\neg C_{\mathrm{completion}})\\
&\lor
(s_{\mathrm{LC}}\geq3.25
\land E_{\mathrm{probable}}),
\end{aligned}
$$

where $s_{\mathrm{LC}}$ is the deterministic event score, $C_{\mathrm{completion}}$ indicates completion censoring, and $E_{\mathrm{probable}}$ denotes the enabled probable-event mode.

### `mergesInFrontOf` and `mergesBehind`

Let $T$ denote the target stream of an already decoded lane change. The shared merge condition is

$$
\begin{aligned}
\mathcal M_{\mathrm{base}}(S,O,T)\mathrel{:\!\Leftrightarrow}{}&
\texttt{changesLane}(S,T)
\land\mathcal M_{\mathrm{pre}}\\
&\land\mathcal M_{\mathrm{path}}
\land\mathcal M_{\mathrm{flow}}
\land\mathcal M_{\mathrm{stable}} .
\end{aligned}
$$

$\mathcal M_{\mathrm{pre}}$ requires $O$ to be present in the target stream before entry, using a $2.0\,\mathrm{s}$ pre-existence window and at least one supporting frame. $\mathcal M_{\mathrm{path}}$ requires target-path overlap at least $0.20$ and at most six topology hops. $\mathcal M_{\mathrm{flow}}$ requires travel-direction difference at most $0.55\,\mathrm{rad}$. $\mathcal M_{\mathrm{stable}}$ requires stable order for at least two frames within a $2.0\,\mathrm{s}$ confirmation window.

Let $O_{\mathrm{rear}}^{*}$ and $O_{\mathrm{front}}^{*}$ be the nearest valid target-stream neighbors behind and ahead of $S$. Then

$$
\begin{aligned}
&\texttt{mergesInFrontOf}(S,O)\\
&\quad\mathrel{:\!\Leftrightarrow}
\mathcal M_{\mathrm{base}}(S,O,T)
\land S\succ_T O\\
&\qquad\land O=O_{\mathrm{rear}}^{*}
\land0<g_{SO}\leq40\,\mathrm{m}\\
&\qquad\land h_{\mathrm{rear}}\leq5.0\,\mathrm{s},
\end{aligned}
$$

and

$$
\begin{aligned}
&\texttt{mergesBehind}(S,O)\\
&\quad\mathrel{:\!\Leftrightarrow}
\mathcal M_{\mathrm{base}}(S,O,T)
\land S\prec_T O\\
&\qquad\land O=O_{\mathrm{front}}^{*}
\land0<g_{SO}\leq40\,\mathrm{m}\\
&\qquad\land h_{\mathrm{front}}\leq5.0\,\mathrm{s}.
\end{aligned}
$$

Here $S\succ_T O$ and $S\prec_T O$ denote stable ahead/behind order on the target path, and $h$ is the corresponding post-merge headway when defined.

### `crossesInFrontOf`

Let $\mathbf d_S$ and $\mathbf d_O$ be unit vectors along the current headings. The two finite forward rays are

$$
\begin{aligned}
R_S(\lambda)&=\mathbf p_S+\lambda\mathbf d_S,
&0\leq\lambda\leq10\,\mathrm{m},\\
R_O(\mu)&=\mathbf p_O+\mu\mathbf d_O,
&0\leq\mu\leq10\,\mathrm{m}.
\end{aligned}
$$

For their unique finite-segment intersection $C$, the implemented relation is

$$
\begin{aligned}
&\texttt{crossesInFrontOf}(S,O)\\
&\quad\mathrel{:\!\Leftrightarrow}
\mathcal X_{\mathrm{type}}
\land\mathcal X_{\mathrm{hist}}
\land\mathcal X_{\mathrm{local}}\\
&\qquad\land\mathcal X_{\mathrm{angle}}
\land\mathcal X_{\mathrm{ray}}
\land\mathcal X_{\mathrm{order}}\\
&\qquad\land\mathcal X_{\mathrm{map}} .
\end{aligned}
$$

$\mathcal X_{\mathrm{type}}$ requires distinct dynamic road users and rejects parked entities; pedestrian–pedestrian pairs are disabled by default. $\mathcal X_{\mathrm{hist}}$ requires at least three observations, simultaneous observation for at least $1.0\,\mathrm{s}$, and track displacement at least $0.30\,\mathrm{m}$. Locality requires $d_c(S,O)\leq30\,\mathrm{m}$, and

$$
\mathcal X_{\mathrm{angle}}
\mathrel{:\!\Leftrightarrow}
25^\circ\leq\Delta\theta_{SO}\leq155^\circ .
$$

The finite-ray condition is

$$
\mathcal X_{\mathrm{ray}}
\mathrel{:\!\Leftrightarrow}
C=R_S(\lambda_C)=R_O(\mu_C),
\quad
\lambda_C,\mu_C\in[0,10]\,\mathrm{m}.
$$

The subject speed must satisfy $v_S\geq0.30\,\mathrm{m/s}$. For a moving object,

$$
t_S(C)=\lambda_C/v_S,\qquad
t_O(C)=\mu_C/v_O,
$$

and crossing order requires

$$
\mathcal X_{\mathrm{order}}
\mathrel{:\!\Leftrightarrow}
0.25\,\mathrm{s}
\leq t_O(C)-t_S(C)
\leq6.0\,\mathrm{s}.
$$

For an effectively stopped object, $t_O(C)$ is treated as unbounded after the parking filter.

The map/context filter uses a $12\,\mathrm{m}$ query radius, maximum road distance $6\,\mathrm{m}$, crosswalk-near distance $5\,\mathrm{m}$, and bicycle road-distance threshold $3\,\mathrm{m}$. Parked-object rejection uses speed at most $0.50\,\mathrm{m/s}$ for at least $2.0\,\mathrm{s}$, displacement at most $1.0\,\mathrm{m}$, and lane distance at most $1.5\,\mathrm{m}$. The conflict footprint/side buffer is $0.35\,\mathrm{m}$ and the side-search window is $3.0\,\mathrm{s}$.

### `yieldsTo`

The object must already be verified to cross in front of the subject. The complete rule is

$$
\begin{aligned}
\texttt{yieldsTo}(S,O)\mathrel{:\!\Leftrightarrow}{}&
\texttt{crossesInFrontOf}(O,S)
\land\mathcal Y_{\mathrm{type}}\\
&\land\mathcal Y_{\mathrm{compete}}
\land\mathcal Y_{\mathrm{concede}}\\
&\land\mathcal Y_{\mathrm{clear}}
\land\mathcal Y_{\mathrm{outside}}\\
&\land\mathcal Y_{\mathrm{proceed}}
\land\neg\mathcal Y_{\mathrm{red}} .
\end{aligned}
$$

The subject is vehicle-like and the object is a dynamic road user. The implementation uses a $4.0\,\mathrm{s}$ pre-event window and $4.0\,\mathrm{s}$ post-event verification window. Before concession, both users must approach the conflict with subject speed at least $0.50\,\mathrm{m/s}$, object speed at least $0.30\,\mathrm{m/s}$, and initial ETA difference

$$
|\tau_S-\tau_O|\leq2.5\,\mathrm{s}.
$$

Let $v_b$ be the subject pre-response baseline speed. It must satisfy $v_b\geq1.0\,\mathrm{m/s}$ and the required speed drop is

$$
\Delta v_{\mathrm{req}} =
\max(0.75,\\;0.15v_b)\quad[\mathrm{m/s}].
$$

The subject must remain within $18\,\mathrm{m}$ of the conflict and outside the conflict while $O$ clears it. Clearance uses half the object length plus a $0.25\,\mathrm{m}$ buffer. A relevant RED traffic signal invalidates the yield interpretation. After clearance,

$$
\begin{aligned}
\mathcal Y_{\mathrm{proceed}}\mathrel{:\!\Leftrightarrow}{}&
v_S^{post}\geq0.80\,\mathrm{m/s}\\
&\land D_S^{post}\geq1.50\,\mathrm{m}.
\end{aligned}
$$

The implementation also records a slow-state reference threshold of $1.25\,\mathrm{m/s}$ when evaluating the concession.

### `overtakes`

Overtaking is a complete, fully observed same-direction Case-1 maneuver:

$$
\begin{aligned}
\texttt{overtakes}(S,O)\mathrel{:\!\Leftrightarrow}{}&
\mathcal O_{\mathrm{hist}}
\land\mathcal O_{\mathrm{start}}
\land\mathcal O_{\mathrm{flow}}\\
&\land\mathcal O_{\mathrm{depart}}
\land\mathcal O_{\mathrm{pass}}
\land\mathcal O_{\mathrm{reverse}}\\
&\land\mathcal O_{\mathrm{return}}
\land\mathcal O_{\mathrm{duration}} .
\end{aligned}
$$

$\mathcal O_{\mathrm{hist}}$ requires at least five common frames, at least $0.5\,\mathrm{s}$ pre-departure history, and frame gaps no larger than $1.5\,\mathrm{s}$. The pair search radius is $80\,\mathrm{m}$. $\mathcal O_{\mathrm{start}}$ requires the same logical corridor and initial subject-behind gap at least $1.0\,\mathrm{m}$.

Same-flow consistency requires

$$
\begin{aligned}
\mathcal O_{\mathrm{flow}}\mathrel{:\!\Leftrightarrow}{}&
\Delta\theta^{travel}_{SO}\leq0.55\,\mathrm{rad}\\
&\land f_{\mathrm{sameFlow}}\geq0.80 .
\end{aligned}
$$

During the passing phase,

$$
\begin{aligned}
\mathcal O_{\mathrm{pass}}\mathrel{:\!\Leftrightarrow}{}&
v_S\geq2.0\,\mathrm{m/s}\\
&\land v_O\geq0.50\,\mathrm{m/s}\\
&\land v_{SO}^{rel,long,+}\geq0.30\,\mathrm{m/s}\\
&\land |r_{SO}|_{\mathrm{side}}\geq1.25\,\mathrm{m}.
\end{aligned}
$$

The departure must use an official/validated adjacent passing corridor. Lane search uses radius $8.0\,\mathrm{m}$, minimum overlap ratio $0.10$, ambiguity margin $0.05$, and at most eight topology hops. Order reversal must produce at least $1.0\,\mathrm{m}$ full clearance, the subject must return stably to the original corridor for two frames, and

$$
\mathcal O_{\mathrm{duration}}
\mathrel{:\!\Leftrightarrow}T_{\mathrm{overtake}}\leq30\,\mathrm{s}.
$$

The approach, passing, and completion phases each require at least $0.5\,\mathrm{s}$.

## Traffic-Control Predicates

#### `controls`

Let $\mathcal M(s)$ denote the controlled lane-connector movements associated with traffic signal $s$. The relation is defined by

$$
\texttt{controls}(s,m)\mathrel{:\!\Leftrightarrow}m\in\mathcal M(s).
$$

The relation is independent of the current signal color.

#### `hasSignalState`

At time $t$, the predicate stores the latest valid traffic-light state $q$ for signal $s$:

$$
\texttt{hasSignalState}(s,q)\mathrel{:\!\Leftrightarrow}q=Q_s(t).
$$

#### `isRelevantSignal`

At time $t$, a traffic signal is relevant to agent $a$ only before entry into the controlled movement:

$$
\begin{aligned}
&\texttt{isRelevantSignal}(a,s)\\
&\quad\mathrel{:\!\Leftrightarrow}\exists m:\texttt{controls}(s,m)\\
&\qquad\land\operatorname{approaches}(a,m,t)\\
&\qquad\land\neg\operatorname{occupies}(a,m,t).
\end{aligned}
$$

The approach and occupancy terms refer to the implementation’s current lane/connector path context; they do not introduce unreported numerical parameters. The predicate stops being emitted once the agent occupies the controlled connector.

## Risk Predicates

#### `hasConflictRiskWith`

The emission condition can be summarized as

$$
\begin{aligned}
&\texttt{hasConflictRiskWith}(S,O)\mathrel{:\!\Leftrightarrow}
\mathcal R_{\mathrm{dyn}}\land\mathcal R_{\mathrm{geom}}\\
&\qquad\land\mathcal R_{\mathrm{closing}}
\land\mathcal R_{\mathrm{interact}}
\land\mathcal R_{\mathrm{broad}}
\land\mathcal R_{\mathrm{future}}
\land\mathcal R_{\mathrm{safety}}.
\end{aligned}
$$

$\mathcal R_{\mathrm{dyn}}$ requires two dynamic road users. $\mathcal R_{\mathrm{geom}}$ requires valid oriented footprints and excludes already-overlapping pairs. $\mathcal R_{\mathrm{closing}}$ requires radial closing speed at least $0.50\,\mathrm{m/s}$. $\mathcal R_{\mathrm{interact}}$ is satisfied by same/connected lane-path topology, a VRU pair within $30\,\mathrm{m}$, a tight same-flow corridor ($|r^{travel}_{SO}|\leq2.5\,\mathrm{m}$, $d_c\leq45\,\mathrm{m}$, direction difference $\leq15^\circ$), or local cross/opposite motion ($d_c\leq40\,\mathrm{m}$ and direction difference $\geq25^\circ$).

Let $r_S$ and $r_O$ be footprint half-diagonals and $d_{\mathrm{CPA}}^{center}$ the minimum center distance under the same constant-velocity model. The broad-phase condition is

$$
\mathcal R_{\mathrm{broad}}
\mathrel{:\!\Leftrightarrow}
d_{\mathrm{CPA}}^{center}
\leq r_S+r_O+0.50\,\mathrm{m}.
$$

If $\tau^{*}$ and $d_{\mathrm{CPA}}^{fs}$ are the time and oriented-footprint clearance at the predicted minimum, then

$$
\begin{aligned}
\mathcal R_{\mathrm{future}}\mathrel{:\!\Leftrightarrow}{}&
0<\tau^{*}\leq H\\
&\land d_{\mathrm{CPA}}^{fs}\leq0.50\,\mathrm{m}\\
&\land d_{fs}(S,O)-d_{\mathrm{CPA}}^{fs}\geq1.0\,\mathrm{m},
\end{aligned}
$$

where $H$ is the configured prediction horizon.

The condition $\mathcal R_{\mathrm{safety}}$ requires the predicted conflict to satisfy the applicable safety criteria described in the Safety-Related Thresholds section. An imminent collision condition is identified when the required collision-avoidance braking demand is at least $5\,\mathrm{m/s^2}$. For lane-change-related conflicts, $\mathcal R_{\mathrm{safety}}$ additionally considers whether the target-lane vehicle would be required to decelerate by more than $3.0\,\mathrm{m/s^2}$ or whether the inter-vehicle gap would fall below the distance travelled by the ALKS vehicle in $1.0\,\mathrm{s}$.

The prediction assumes constant velocity and fixed heading. The base predicate implementation uses a $2.0\,\mathrm{s}$ horizon; the retained standard launcher passes $H=3.0\,\mathrm{s}$. The risk computation uses a $0.50\,\mathrm{m}$ predicted-clearance threshold, minimum radial closing speed $0.50\,\mathrm{m/s}$, minimum clearance reduction $1.0\,\mathrm{m}$, and 24 bounded optimization iterations. The relation is emitted once per unordered pair and frame and does not use observed future ground truth.
